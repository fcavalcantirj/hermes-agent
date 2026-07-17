"""zvec-memory core: indexer + search over Hermes conversation history.

Library layer — no `mcp` dependency, testable alone (SPEC §Core).
Read-only over state.db; owns a derived zvec collection that is always
rebuildable from state.db, so auto-creating it is safe (the opposite of the
SSOT rule, *because* this store is derived, never authoritative).

Dual-vector collection `hermes_memory`:
  v_local (384, Flat/COSINE)  — fastembed bge-small-en-v1.5, sovereignty lane
  v_jina  (2048, Flat/COSINE) — jina-embeddings-v4, quality lane
zvec scores are cosine DISTANCES (ascending; spike-verified 2026-07-16).
zvec collections are single-owner (second open raises "Can't lock") — every
collection open in this module happens under the flock in `_locked`.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sqlite3
import time
import urllib.request
from pathlib import Path

import zvec

COLLECTION_NAME = "hermes_memory"
LOCAL_DIM = 384
JINA_DIM = 2048
SNIPPET_MAX = 300
MIN_CONTENT_LEN = 12
JINA_BATCH_MAX = 128
LOCK_TIMEOUT_S = 20.0  # tests shrink this; poll interval stays fixed
_LOCK_POLL_S = 0.2

JINA_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v4"
LOCAL_MODEL = "BAAI/bge-small-en-v1.5"

_SELECT_NEW_ROWS = (
    "SELECT id, session_id, role, content, timestamp FROM messages "
    "WHERE id > :wm AND role IN ('user','assistant') AND content IS NOT NULL "
    "AND length(content) >= :minlen AND active = 1 ORDER BY id LIMIT :limit"
)
_COUNT_REMAINING = (
    "SELECT COUNT(*) FROM messages "
    "WHERE id > :wm AND role IN ('user','assistant') AND content IS NOT NULL "
    "AND length(content) >= :minlen AND active = 1"
)


class StateDBUnavailable(Exception):
    """state.db missing or unreadable — explicit, never an empty success."""


class StoreBusy(Exception):
    """flock not acquired within LOCK_TIMEOUT_S — caller retries."""


# ---------------------------------------------------------------- env / paths


def _memory_dir() -> Path:
    return Path(os.environ.get("ZVEC_MEMORY_DIR", "~/.hermes/zvec-memory")).expanduser()


def _state_db_path() -> Path:
    return Path(os.environ.get("ZVEC_MEMORY_STATE_DB", "~/.hermes/state.db")).expanduser()


def _jina_key_file() -> Path:
    default = _memory_dir() / "jina.key"
    return Path(os.environ.get("ZVEC_MEMORY_JINA_KEY_FILE", str(default))).expanduser()


# ------------------------------------------------------------- embedder seam
# Module-level functions on purpose: tests monkeypatch
# `zvec_memory_core.embed_local` / `embed_jina` and every internal call goes
# through the module global, so the patch always takes.

_local_model = None


def embed_local(texts: list[str]) -> list[list[float]]:
    """Lazy fastembed singleton (first real call downloads the ONNX model)."""
    global _local_model
    if _local_model is None:
        from fastembed import TextEmbedding  # deferred: import cost + optionality

        _local_model = TextEmbedding(model_name=LOCAL_MODEL)
    return [[float(x) for x in vec] for vec in _local_model.embed(list(texts))]


def embed_jina(texts: list[str], task: str) -> list[list[float]]:
    """POST to the Jina API. Raises on ANY failure — the caller degrades."""
    key_file = _jina_key_file()
    key = key_file.read_text().strip()
    body = json.dumps(
        {"model": JINA_MODEL, "task": task, "input": [{"text": t} for t in texts]}
    ).encode("utf-8")
    req = urllib.request.Request(
        JINA_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            # bare python-urllib UA gets 403 — browser-ish UA required
            "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) zvec-memory/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    data = sorted(payload["data"], key=lambda d: d["index"])
    vecs = [d["embedding"] for d in data]
    if len(vecs) != len(texts) or any(len(v) != JINA_DIM for v in vecs):
        raise RuntimeError(
            f"jina returned {len(vecs)} vectors (wanted {len(texts)}) "
            f"or wrong dimension (wanted {JINA_DIM})"
        )
    return vecs


# ------------------------------------------------------------------- locking


@contextlib.contextmanager
def _locked(mem_dir: Path, timeout_s: float | None = None):
    """flock LOCK_EX with a poll-timeout. zvec collections are single-owner,
    so EVERY collection open (index, backfill, search, status) sits inside."""
    if timeout_s is None:
        timeout_s = LOCK_TIMEOUT_S
    mem_dir.mkdir(parents=True, exist_ok=True)
    lock_path = mem_dir / "lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise StoreBusy(
                        f"lock at {lock_path} still held after {timeout_s:.0f}s"
                    ) from None
                time.sleep(_LOCK_POLL_S)
        yield
    finally:
        os.close(fd)  # flock drops with the fd


# --------------------------------------------------------- collection / state


def _collection_path(mem_dir: Path) -> Path:
    return mem_dir / "collection"


def _open_collection(mem_dir: Path):
    """Open (or create — safe: derived store) the dual-vector collection.
    Caller MUST hold the flock and drop the returned object promptly."""
    path = _collection_path(mem_dir)
    if path.exists():
        return zvec.open(str(path))
    schema = zvec.CollectionSchema(
        name=COLLECTION_NAME,
        fields=[
            zvec.FieldSchema("session_id", zvec.DataType.STRING),
            zvec.FieldSchema("role", zvec.DataType.STRING),
            zvec.FieldSchema("timestamp", zvec.DataType.STRING),
            zvec.FieldSchema("snippet", zvec.DataType.STRING),
        ],
        vectors=[
            zvec.VectorSchema(
                "v_local",
                zvec.DataType.VECTOR_FP32,
                dimension=LOCAL_DIM,
                index_param=zvec.FlatIndexParam(metric_type=zvec.MetricType.COSINE),
            ),
            zvec.VectorSchema(
                "v_jina",
                zvec.DataType.VECTOR_FP32,
                dimension=JINA_DIM,
                index_param=zvec.FlatIndexParam(metric_type=zvec.MetricType.COSINE),
            ),
        ],
    )
    return zvec.create_and_open(str(path), schema)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_json(path: Path, obj) -> None:
    # temp+rename: a crashed write must not leave truncated JSON behind
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _read_watermark(mem_dir: Path) -> int:
    return int(_read_json(mem_dir / "watermark.json", {"last_id": 0})["last_id"])


def _write_watermark(mem_dir: Path, last_id: int) -> None:
    _write_json(mem_dir / "watermark.json", {"last_id": int(last_id)})


def _read_pending(mem_dir: Path) -> list[str]:
    return list(_read_json(mem_dir / "jina_pending.json", []))


def _write_pending(mem_dir: Path, ids: list[str]) -> None:
    _write_json(mem_dir / "jina_pending.json", ids)


def _raise_on_bad_status(statuses, op: str) -> None:
    # zvec Status exposes ok()/code() as METHODS (pybind11), not attributes —
    # comparing the bound method to 0 is always-truthy (introspected 2026-07-16)
    if statuses is None:
        return
    if not isinstance(statuses, list):
        statuses = [statuses]
    for st in statuses:
        if not st.ok():
            raise RuntimeError(f"zvec {op} failed: {st}")


# ---------------------------------------------------------------- operations


def index_catchup(limit: int = 500) -> dict:
    """Pull new qualifying messages from state.db into the collection.

    Watermark advances ONLY after a successful write, so a crashed run
    re-processes the same rows — which is why the write is an upsert
    (a plain insert would fail on the duplicate doc ids of a re-run).
    """
    mem_dir = _memory_dir()
    mem_dir.mkdir(parents=True, exist_ok=True)
    db_path = _state_db_path()
    if not db_path.exists():
        raise StateDBUnavailable(str(db_path))
    try:
        # mode=ro URI: this module must never write state.db
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise StateDBUnavailable(f"{db_path} ({exc})") from exc

    with contextlib.closing(con):
        wm = _read_watermark(mem_dir)
        try:
            rows = con.execute(
                _SELECT_NEW_ROWS,
                {"wm": wm, "minlen": MIN_CONTENT_LEN, "limit": limit},
            ).fetchall()
        except sqlite3.Error as exc:
            raise StateDBUnavailable(f"{db_path} ({exc})") from exc

        pending = _read_pending(mem_dir)
        if not rows:
            lag = con.execute(
                _COUNT_REMAINING, {"wm": wm, "minlen": MIN_CONTENT_LEN}
            ).fetchone()[0]
            return {"indexed": 0, "pending_jina": len(pending), "lag": int(lag)}

        # Embed the STORED snippet, not the full content: backfill_jina can
        # only see the snippet later, so index-time must embed the same text
        # or the two lanes drift apart on long messages.
        snippets = [str(r[3])[:SNIPPET_MAX] for r in rows]
        local_vecs = embed_local(snippets)

        jina_vecs: list[list[float] | None] = [None] * len(rows)
        key_present = _jina_key_file().exists()
        if key_present:
            for start in range(0, len(snippets), JINA_BATCH_MAX):
                chunk = snippets[start : start + JINA_BATCH_MAX]
                try:
                    vecs = embed_jina(chunk, "retrieval.passage")
                except Exception:
                    continue  # this chunk stays None -> zero-vector + pending
                for offset, vec in enumerate(vecs):
                    jina_vecs[start + offset] = vec

        docs = []
        newly_pending = []
        for row, lvec, jvec in zip(rows, local_vecs, jina_vecs):
            msg_id, session_id, role, _content, ts = row
            doc_id = f"m{msg_id}"
            if jvec is None:
                jvec = [0.0] * JINA_DIM  # placeholder until backfill_jina
                newly_pending.append(doc_id)
            docs.append(
                zvec.Doc(
                    id=doc_id,
                    vectors={"v_local": lvec, "v_jina": jvec},
                    fields={
                        "session_id": str(session_id),
                        "role": str(role),
                        "timestamp": str(ts),
                        "snippet": str(row[3])[:SNIPPET_MAX],
                    },
                )
            )

        with _locked(mem_dir):
            coll = _open_collection(mem_dir)
            try:
                _raise_on_bad_status(coll.upsert(docs), "upsert")
            finally:
                del coll

        new_wm = rows[-1][0]
        _write_watermark(mem_dir, new_wm)
        for doc_id in newly_pending:
            if doc_id not in pending:
                pending.append(doc_id)
        _write_pending(mem_dir, pending)

        lag = con.execute(
            _COUNT_REMAINING, {"wm": new_wm, "minlen": MIN_CONTENT_LEN}
        ).fetchone()[0]
        return {"indexed": len(docs), "pending_jina": len(pending), "lag": int(lag)}


def backfill_jina(cap: int = 64) -> int:
    """Re-embed up to `cap` pending docs with the real jina lane.
    Failures leave the ids queued (visible via pending_jina); returns count."""
    mem_dir = _memory_dir()
    if not _jina_key_file().exists():
        return 0
    pending = _read_pending(mem_dir)
    if not pending:
        return 0
    batch = pending[:cap]

    with _locked(mem_dir):
        coll = _open_collection(mem_dir)
        try:
            fetched = coll.fetch(batch, include_vector=False)
            ids, texts, gone = [], [], []
            for doc_id in batch:
                doc = fetched.get(doc_id)
                if doc is None:
                    gone.append(doc_id)  # not in the collection -> nothing to fix
                    continue
                ids.append(doc_id)
                texts.append((doc.fields or {}).get("snippet", ""))

            done: list[str] = []
            if ids:
                try:
                    vecs = embed_jina(texts, "retrieval.passage")
                except Exception:
                    vecs = None  # every id stays queued; count says 0
                if vecs is not None:
                    for doc_id, vec in zip(ids, vecs):
                        try:
                            _raise_on_bad_status(
                                coll.update(zvec.Doc(id=doc_id, vectors={"v_jina": vec})),
                                "update",
                            )
                        except Exception:
                            continue  # this id stays queued
                        done.append(doc_id)
        finally:
            del coll

    remove = set(done) | set(gone)
    if remove:
        _write_pending(mem_dir, [p for p in pending if p not in remove])
    return len(done)


def search(query: str, top_k: int = 5, lane: str = "auto") -> dict:
    """Semantic search. `auto` prefers jina when the key file exists and
    DEGRADES LOUDLY to local on any jina failure (never a silent switch)."""
    if lane not in ("auto", "local", "jina"):
        raise ValueError(f"unknown lane {lane!r} (want auto|local|jina)")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    mem_dir = _memory_dir()
    key_present = _jina_key_file().exists()
    if lane == "jina" and not key_present:
        raise RuntimeError(f"jina lane forced but key file absent: {_jina_key_file()}")

    degraded = None
    field = "v_local"
    qvec = None
    if lane == "jina" or (lane == "auto" and key_present):
        try:
            qvec = embed_jina([query], "retrieval.query")[0]
            field = "v_jina"
        except Exception as exc:
            if lane == "jina":
                raise RuntimeError(f"jina lane forced but embedding failed: {exc}") from exc
            degraded = f"jina-unavailable: {exc}"
    if qvec is None:
        qvec = embed_local([query])[0]
        field = "v_local"

    pending = set(_read_pending(mem_dir))

    # Embedding happened OUTSIDE the lock (jina takes seconds); only the
    # single-owner collection open sits inside.
    with _locked(mem_dir):
        coll = _open_collection(mem_dir)
        try:
            hits = coll.query(
                queries=zvec.Query(field_name=field, vector=qvec), topk=top_k
            )
            doc_count = coll.stats.doc_count
        finally:
            del coll

    results = []
    for doc in hits:
        if field == "v_jina" and doc.id in pending:
            continue  # zero-vector placeholder, not a real jina embedding
        f = doc.fields or {}
        results.append(
            {
                "message_id": int(doc.id[1:]),  # doc id = m<message_rowid>
                "session_id": f.get("session_id"),
                "role": f.get("role"),
                "timestamp": f.get("timestamp"),
                "snippet": f.get("snippet"),
                "distance": doc.score,  # cosine distance, ascending = closer
            }
        )

    out = {
        "success": True,
        "lane": "jina" if field == "v_jina" else "local",
        "count": len(results),
        "results": results,
        # index block makes staleness visible: a 0-count with a healthy
        # index is an HONEST zero, not a failure
        "index": {
            "watermark": _read_watermark(mem_dir),
            "pending_jina": len(pending),
            "doc_count": int(doc_count),
        },
    }
    if degraded:
        out["degraded"] = degraded
    return out


def index_status() -> dict:
    """Cheap health view — no indexing, no embedding."""
    mem_dir = _memory_dir()
    db_path = _state_db_path()
    doc_count = 0
    if _collection_path(mem_dir).exists():
        with _locked(mem_dir):
            coll = zvec.open(str(_collection_path(mem_dir)))
            try:
                doc_count = int(coll.stats.doc_count)
            finally:
                del coll
    return {
        "success": True,
        "watermark": _read_watermark(mem_dir),
        "doc_count": doc_count,
        "pending_jina": len(_read_pending(mem_dir)),
        "state_db": {"path": str(db_path), "exists": db_path.exists()},
        "lanes": {"local": True, "jina": _jina_key_file().exists()},
    }
