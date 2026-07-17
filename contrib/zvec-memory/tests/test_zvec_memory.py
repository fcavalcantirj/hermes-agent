"""Unit tests for zvec_memory_core — the derived semantic-recall index over state.db.

Constraints honored here (SPEC "Tests: no network, no model download"):
- Embedders are faked via monkeypatch: deterministic hash-vectors of the correct
  dims (384 local, 2048 jina, pinned by the SPEC's Collection contract). Nothing
  downloads a model or touches the network.
- state.db is a tmp SQLite built from the minimal messages/sessions DDL the SPEC's
  SELECT reads; the core opens it strictly read-only, so a plain file suffices.
- Domain is coding-only in every fixture string (SPEC house rule).
- Each case is red-on-demand: it guards a failure that must stay VISIBLE — missing
  DB, lock contention, stale watermark, honest zero, jina degrade, row filtering —
  never a silent empty-success.

The core module may not exist when this file is authored; it is imported by name
(conftest.py puts the parent dir on sys.path) and exercised only through the public
contract the SPEC names: index_catchup / backfill_jina / search / StateDBUnavailable
/ StoreBusy / embed_local / embed_jina.
"""

import hashlib
import math
import sqlite3
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import zvec_memory_core as core

# Vector dims are fixed by the collection schema (SPEC "Collection": v_local 384,
# v_jina 2048). Pin them here rather than reading a core constant so a fake that
# returns the wrong length is caught regardless of how the core names its dims.
DIM_LOCAL = 384
DIM_JINA = 2048


# --------------------------------------------------------------------------- #
# Deterministic offline embedders (correct dims, unit-normalized for COSINE).
# --------------------------------------------------------------------------- #
def _hash_vec(text, dim):
    """Stable pseudo-random unit vector from `text`: identical text ⇒ identical
    vector, so watermark idempotency and distance ordering are reproducible."""
    vals = []
    counter = 0
    while len(vals) < dim:
        digest = hashlib.sha256(("%d:%s" % (counter, text)).encode("utf-8")).digest()
        for i in range(0, len(digest), 4):
            if len(vals) >= dim:
                break
            (u,) = struct.unpack("<I", digest[i:i + 4])
            vals.append((u / 4294967296.0) * 2.0 - 1.0)  # -> [-1, 1)
        counter += 1
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def fake_local(texts):
    # Mirrors embed_local(texts) -> list[vec384]; the "local:" prefix is applied to
    # BOTH passages and queries, so identical query/content text ⇒ distance 0.
    return [_hash_vec("local:" + t, DIM_LOCAL) for t in texts]


def fake_jina(texts, task=None):
    # Mirrors embed_jina(texts, task) -> list[vec2048]; task folds into the seed so
    # retrieval.passage and retrieval.query embeddings differ, as with the real
    # task-adapted model (⇒ no exact-match shortcut is asserted on the jina lane).
    prefix = "jina:%s:" % (task or "")
    return [_hash_vec(prefix + t, DIM_JINA) for t in texts]


def boom_jina(texts, task=None):
    # Simulates any jina failure (403/timeout/5xx). The caller MUST degrade, not crash.
    raise RuntimeError("simulated jina 503 upstream")


# --------------------------------------------------------------------------- #
# state.db construction — minimal DDL matching the SPEC SELECT.
# --------------------------------------------------------------------------- #
# SPEC index_catchup SELECT:
#   SELECT id, session_id, role, content, timestamp FROM messages
#   WHERE id > :wm AND role IN ('user','assistant') AND content IS NOT NULL
#     AND length(content) >= 12 AND active = 1 ORDER BY id LIMIT :limit
_DDL = """
CREATE TABLE sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    created_at TEXT
);
CREATE TABLE messages (
    id         INTEGER PRIMARY KEY,
    session_id TEXT,
    role       TEXT,
    content    TEXT,
    timestamp  TEXT,
    active     INTEGER NOT NULL DEFAULT 1
);
"""


def _build_state_db(path, rows):
    """rows: iterable of dict(id, session_id, role, content, timestamp, active)."""
    rows = list(rows)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_DDL)
        seen = {}
        for r in rows:
            seen.setdefault(r["session_id"], r["timestamp"])
        conn.executemany(
            "INSERT INTO sessions(id, title, created_at) VALUES (?,?,?)",
            [(sid, "session " + sid, ts) for sid, ts in seen.items()],
        )
        conn.executemany(
            "INSERT INTO messages(id, session_id, role, content, timestamp, active)"
            " VALUES (:id,:session_id,:role,:content,:timestamp,:active)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


# Sentinel so an explicit content=None stores a real SQL NULL, distinct from
# "caller omitted content, use the default". Without this, content=None would be
# indistinguishable from the default and the NULL-content filter case would not
# actually exercise a NULL (a silent false-pass).
_DEFAULT_CONTENT = object()


def _row(mid, role="user", content=_DEFAULT_CONTENT, active=1, session_id="s-refactor"):
    # Coding-domain content, >= 12 chars by default (passes the length filter).
    if content is _DEFAULT_CONTENT:
        content = "message %d: refactor the goroutine pool and add a context deadline" % mid
    return {
        "id": mid,
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": "2026-07-1%d 09:%02d:00" % (mid % 10, mid % 60),
        "active": active,
    }


def _valid_rows(n, start=1):
    prompts = [
        "How do I cancel a goroutine with context.Context in Go?",
        "Wrap the flock acquire in a timeout and raise StoreBusy on expiry.",
        "The migrate step should be idempotent and re-run cleanly.",
        "Use bge-small ONNX for the local embedding lane, 384 dims.",
        "Add a watermark so index_catchup only reads new message ids.",
        "Return an honest zero with the index block, never a bare empty.",
    ]
    role = ("user", "assistant")
    return [
        _row(start + i, role=role[i % 2], content=prompts[i % len(prompts)],
             session_id="s-%d" % (i % 3))
        for i in range(n)
    ]


def _mid_int(message_id):
    """Normalize a returned message_id ('m42' or 42) to the int rowid."""
    s = str(message_id)
    return int(s[1:]) if s[:1] in ("m", "M") else int(s)


def _assert_index_block(res):
    assert "index" in res, "search result must carry an index block to expose staleness"
    idx = res["index"]
    for key in ("watermark", "pending_jina", "doc_count"):
        assert key in idx, "index block missing %r" % key


# --------------------------------------------------------------------------- #
# Fixture: tmp dir + env + faked embedders. Env is read per-call by the core, so
# setenv here governs every subsequent core call in the test.
# --------------------------------------------------------------------------- #
@pytest.fixture
def mem(tmp_path, monkeypatch):
    d = tmp_path / "zmem"
    d.mkdir()
    state_db = tmp_path / "state.db"
    key_path = d / "jina.key"
    monkeypatch.setenv("ZVEC_MEMORY_DIR", str(d))
    monkeypatch.setenv("ZVEC_MEMORY_STATE_DB", str(state_db))
    monkeypatch.setenv("ZVEC_MEMORY_JINA_KEY_FILE", str(key_path))
    # Keep the lock-contention case fast IF the core honors a timeout knob; harmless
    # otherwise (the real 20 s poll still raises StoreBusy, just slower — the lock
    # holder below outlives even that).
    monkeypatch.setenv("ZVEC_MEMORY_LOCK_TIMEOUT", "1")
    for name in ("LOCK_TIMEOUT", "LOCK_TIMEOUT_S", "STORE_LOCK_TIMEOUT", "LOCK_POLL_TIMEOUT"):
        monkeypatch.setattr(core, name, 1, raising=False)
    monkeypatch.setattr(core, "embed_local", fake_local)
    monkeypatch.setattr(core, "embed_jina", fake_jina)
    return SimpleNamespace(dir=d, state_db=state_db, key_path=key_path, mp=monkeypatch)


# =========================================================================== #
# Case 1 — missing state.db → explicit error (never empty success).
# =========================================================================== #
def test_case1_missing_state_db_raises(mem):
    # Fixture never created the file: the path is valid but nonexistent.
    assert not mem.state_db.exists()
    with pytest.raises(core.StateDBUnavailable):
        core.index_catchup()


# =========================================================================== #
# Case 2 — lock held by another process → StoreBusy within the timeout.
# =========================================================================== #
_LOCK_CHILD = (
    "import sys, os, fcntl, time\n"
    "lock_path, sentinel = sys.argv[1], sys.argv[2]\n"
    "f = open(lock_path, 'w')\n"
    "fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"      # hold the exclusive advisory lock
    "open(sentinel, 'w').close()\n"                 # signal 'lock acquired'
    "time.sleep(30)\n"                              # outlive even the full 20 s poll
)


def test_case2_lock_held_raises_storebusy(mem, tmp_path):
    _build_state_db(mem.state_db, _valid_rows(3))
    lock_path = mem.dir / "lock"
    lock_path.write_text("")  # ensure the inode exists before the child opens it
    sentinel = tmp_path / "lock.acquired"

    child = subprocess.Popen([sys.executable, "-c", _LOCK_CHILD, str(lock_path), str(sentinel)])
    try:
        # Contend only once the child actually holds the flock.
        deadline = time.time() + 5.0
        while not sentinel.exists() and time.time() < deadline:
            if child.poll() is not None:
                raise AssertionError("lock-holder child exited early")
            time.sleep(0.02)
        assert sentinel.exists(), "child never acquired the lock"

        with pytest.raises(core.StoreBusy):
            core.index_catchup()
    finally:
        child.terminate()
        child.wait(timeout=10)


# =========================================================================== #
# Case 3 — watermark idempotency: second index_catchup indexes 0.
# =========================================================================== #
def test_case3_watermark_idempotent(mem):
    _build_state_db(mem.state_db, _valid_rows(4))
    first = core.index_catchup()
    for key in ("indexed", "pending_jina", "lag"):
        assert key in first, "index_catchup must return %r" % key
    assert first["indexed"] == 4
    second = core.index_catchup()
    assert second["indexed"] == 0, "watermark must gate already-indexed rows"
    assert second["lag"] == 0, "nothing left to catch up ⇒ zero lag"


# =========================================================================== #
# Case 4 — honest zero: healthy index, nothing to match → success true, count 0.
# =========================================================================== #
def test_case4_honest_zero(mem):
    # All rows are 'system' ⇒ filtered out, so index_catchup opens/creates the
    # collection (create_and_open — SPEC derived-store auto-create) yet inserts 0.
    # The index is therefore healthy (reachable, watermarked) but genuinely empty.
    rows = [_row(1, role="system"), _row(2, role="system")]
    _build_state_db(mem.state_db, rows)
    caught = core.index_catchup()
    assert caught["indexed"] == 0

    res = core.search("where is the release script generated", top_k=5, lane="local")
    assert res["success"] is True, "an empty-but-healthy index is an honest zero, not a failure"
    assert res["count"] == 0
    assert res["results"] == []
    _assert_index_block(res)
    assert res["index"]["doc_count"] == 0


# =========================================================================== #
# Case 5 — no key → all docs pending; then jina appears → backfill drains queue.
# =========================================================================== #
def test_case5_no_key_pending_then_backfill_drains(mem):
    _build_state_db(mem.state_db, _valid_rows(3))
    assert not mem.key_path.exists()  # local-only mode: no key file

    caught = core.index_catchup()
    assert caught["indexed"] == 3
    assert caught["pending_jina"] == 3, "with no key every doc must queue for jina backfill"

    # Keyless backfill is a no-op (nothing to embed against).
    assert core.backfill_jina(cap=64) == 0

    # Jina becomes available: key file appears; embedder was already faked-working.
    mem.key_path.write_text("fake-key-not-used-by-the-embedder-seam")
    assert core.backfill_jina(cap=64) == 3, "backfill must embed and clear the whole queue"

    res = core.search("goroutine deadline", top_k=5, lane="local")
    _assert_index_block(res)
    assert res["index"]["pending_jina"] == 0, "queue must be empty after a full backfill"


# =========================================================================== #
# Case 6 — results carry metadata + ascending distances; lane reflects reality.
# Parametrized over (forced lane, key present): this is the parametrization proof.
# =========================================================================== #
@pytest.mark.parametrize(
    "lane,key_present,expected_lane",
    [
        ("local", False, "local"),   # sovereignty lane, no key needed
        ("local", True, "local"),    # explicit local ignores the key
        ("auto", False, "local"),    # auto with no key ⇒ local floor
        ("auto", True, "jina"),      # auto with key ⇒ quality lane
        ("jina", True, "jina"),      # forced jina, key present
    ],
)
def test_case6_metadata_and_ascending_distance(mem, lane, key_present, expected_lane):
    _build_state_db(mem.state_db, _valid_rows(6))
    if key_present:
        mem.key_path.write_text("fake-key")
    core.index_catchup()

    res = core.search("cancel the goroutine with a context deadline", top_k=5, lane=lane)
    assert res["success"] is True
    assert res["lane"] == expected_lane, "lane must reflect the real path taken, not a wish"
    assert "degraded" not in res, "a working lane must not report degradation"
    assert res["count"] == len(res["results"])
    assert res["count"] > 0
    _assert_index_block(res)

    valid_sessions = {"s-0", "s-1", "s-2"}
    distances = []
    for r in res["results"]:
        for field in ("message_id", "session_id", "role", "timestamp", "snippet", "distance"):
            assert field in r, "result missing metadata field %r" % field
        assert r["role"] in ("user", "assistant")
        assert r["session_id"] in valid_sessions
        assert isinstance(r["snippet"], str) and r["snippet"]
        distances.append(r["distance"])
    # zvec cosine scores are DISTANCES, ascending (nearest first) — SPEC.
    assert distances == sorted(distances), "distances must be non-decreasing (nearest first)"


# --------------------------------------------------------------------------- #
# Load-bearing companion to Case 6 — an EXACT query/content match must rank
# first on the local lane (deterministic: identical text ⇒ distance 0). Catches a
# result set that returns rows but in the wrong (or unsorted) order.
# --------------------------------------------------------------------------- #
def test_local_exact_match_ranks_first(mem):
    needle = "regenerate the parser tables from grammar.y at build time"
    rows = _valid_rows(4) + [_row(99, role="user", content=needle, session_id="s-0")]
    _build_state_db(mem.state_db, rows)
    core.index_catchup()

    res = core.search(needle, top_k=5, lane="local")
    assert res["count"] > 0
    assert _mid_int(res["results"][0]["message_id"]) == 99, "exact match must be nearest"
    assert res["results"][0]["distance"] <= min(r["distance"] for r in res["results"])


# =========================================================================== #
# Case 7 — jina failure mid-search → degrade to local, with a visible reason.
# =========================================================================== #
def test_case7_jina_failure_degrades_to_local(mem):
    _build_state_db(mem.state_db, _valid_rows(4))
    mem.key_path.write_text("fake-key")            # key present ⇒ auto would pick jina...
    mem.mp.setattr(core, "embed_jina", boom_jina)  # ...but jina is down.

    # index still succeeds locally (jina failure ⇒ zero-vector + pending, not a crash).
    core.index_catchup()

    res = core.search("idempotent migrate step", top_k=5, lane="auto")
    assert res["success"] is True, "a jina outage must degrade, never fail the search"
    assert res["lane"] == "local", "must fall back to the local lane"
    assert "degraded" in res, "the degrade must be surfaced, never silent"
    assert res["degraded"].startswith("jina-unavailable"), (
        "degrade reason must name the cause: got %r" % res.get("degraded")
    )
    assert res["count"] == len(res["results"])


# =========================================================================== #
# Case 8 — only role user/assistant, content >= 12, active = 1 rows are indexed.
# =========================================================================== #
def test_case8_filters_only_valid_rows(mem):
    rows = [
        _row(1, role="user"),                        # valid
        _row(2, role="assistant"),                   # valid
        _row(3, role="system"),                      # excluded: role
        _row(4, role="user", content="too short"),   # excluded: len("too short") == 9 < 12
        _row(5, role="user", active=0),              # excluded: inactive
        _row(6, role="assistant", content=None),     # excluded: NULL content
        _row(7, role="user"),                        # valid
    ]
    _build_state_db(mem.state_db, rows)

    caught = core.index_catchup()
    assert caught["indexed"] == 3, "exactly the 3 qualifying rows must be indexed"

    res = core.search("refactor the goroutine pool", top_k=50, lane="local")
    got = {_mid_int(r["message_id"]) for r in res["results"]}
    assert got == {1, 2, 7}, "indexed set must be exactly the valid rowids, got %r" % got


# --------------------------------------------------------------------------- #
# Load-bearing companion to Case 8 — isolate each exclusion reason so a filter
# dropping the wrong predicate is caught individually (parametrized).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_kind,bad_row",
    [
        ("role_system", _row(2, role="system")),
        ("short_content", _row(2, content="only nine!")),   # 10 chars < 12
        ("inactive", _row(2, active=0)),
        ("null_content", _row(2, content=None)),
        ("role_tool", _row(2, role="tool")),                # non user/assistant
    ],
)
def test_filter_excludes_each_reason(mem, bad_kind, bad_row):
    valid = _row(1, role="user")
    _build_state_db(mem.state_db, [valid, bad_row])

    caught = core.index_catchup()
    assert caught["indexed"] == 1, "%s row must be excluded" % bad_kind

    res = core.search("refactor the goroutine pool", top_k=50, lane="local")
    got = {_mid_int(r["message_id"]) for r in res["results"]}
    assert got == {1}, "only the valid row (id 1) should be indexed for %s" % bad_kind
