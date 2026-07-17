# zvec-memory — semantic recall for DasBrowCoder (SPEC, 2026-07-17)

Standalone stdio MCP server adding a **semantic search lane over conversation history**.
Zero hermes-fork changes: the SDK loads user-scope MCP servers (verified: the SDK passes
`--strict-mcp-config` only when the option is set, and our provider doesn't set it).
The FTS `session_search` shim stays untouched as the keyword floor; the native memory-file
system stays the write path. This plugin is **read-only over state.db** and **owns its own
derived index** (rebuildable from state.db ⇒ auto-creating its collection is safe — the
opposite of the SSOT auto-create rule, *because* this store is derived, never authoritative).

Decisions carried from measurement (memory-recall-eval-r1.md):
- Dual-vector: `v_jina` (jina-embeddings-v4, 2048-dim, task-adapted) = quality lane,
  93%/77% hit@5; `v_local` (bge-small-en-v1.5 ONNX, 384-dim) = sovereignty lane, 67%/43%,
  **13 ms/query on the Pi**. Jina from the Pi: 1.5–2.1 s warm ⇒ jina serves explicit
  searches only; anything per-turn rides local.
- zvec collections are **single-owner** (LOCK error on second open — spike-proven) ⇒
  open-per-call under an `flock`, never a long-lived hold.
- zvec API notes: scores are cosine **distances** (ascending); collection names have a
  min-length regex; `VectorQuery` deprecated → use `Query` if present (introspect; fall
  back to `VectorQuery`).

## Files

- `zvec_memory_core.py` — library, no `mcp` dependency (testable alone): indexer + search.
- `zvec_memory_server.py` — FastMCP stdio server exposing the two tools.
- `tests/test_zvec_memory.py` — pytest, no network, embedder seam faked.
- `README.md` — Pi install/register/allowlist/red-on-demand runbook.
- `deploy.sh` — scp to Pi + sha256 verify both sides (house style).

## Core (`zvec_memory_core.py`)

Env (all optional): `ZVEC_MEMORY_DIR` (default `~/.hermes/zvec-memory`),
`ZVEC_MEMORY_STATE_DB` (default `~/.hermes/state.db`), `ZVEC_MEMORY_JINA_KEY_FILE`
(default `<dir>/jina.key`, mode-600; absent ⇒ local-only mode).

Layout under the dir: `collection/` (zvec), `watermark.json` (`{"last_id": N}`),
`jina_pending.json` (list of doc ids awaiting jina backfill), `lock` (flock file).

Collection: name `hermes_memory`; vectors `v_local` (384, Flat/COSINE) + `v_jina`
(2048, Flat/COSINE); doc id = `m<message_rowid>`. Metadata (session_id, role, timestamp,
snippet ≤300 chars) via zvec `FieldSchema` scalar fields if the API cooperates, else a
sidecar `meta.json` — implementer's choice, but results MUST return the metadata.

Embedder seam (monkeypatchable module functions):
- `embed_local(texts) -> list[vec384]` — lazy fastembed singleton.
- `embed_jina(texts, task) -> list[vec2048]` — urllib POST to `https://api.jina.ai/v1/embeddings`,
  model `jina-embeddings-v4`, `task` = `retrieval.passage`|`retrieval.query`, input as
  `[{"text": t}]`, **browser-ish User-Agent** (bare python-urllib gets 403), key read from
  the key file, 60 s timeout. Raises on any failure (caller degrades).

`index_catchup(limit=500)`:
- Read state.db strictly read-only (`file:<path>?mode=ro` URI):
  `SELECT id, session_id, role, content, timestamp FROM messages WHERE id > :wm AND
  role IN ('user','assistant') AND content IS NOT NULL AND length(content) >= 12 AND
  active = 1 ORDER BY id LIMIT :limit`.
- Missing/unopenable state.db ⇒ raise `StateDBUnavailable` (explicit, never empty).
- Embed batch local (always). Jina: if key file present, try batch ≤128 with
  `retrieval.passage`; on failure or no key ⇒ `v_jina` = zero-vector + id appended to
  pending. Insert docs; write watermark ONLY after successful insert (idempotent re-run).
- Return `{"indexed": n, "pending_jina": len(pending), "lag": remaining_estimate}`.

`backfill_jina(cap=64)`: if key present and pending non-empty: embed ≤cap pending docs'
snippets (`retrieval.passage`), `coll.update`/`upsert` their `v_jina`, drop from pending.
Failures leave ids queued. Returns count.

`search(query, top_k=5, lane="auto")`:
- `auto`: jina when key file present → embed query `retrieval.query`, search `v_jina`;
  ANY jina failure ⇒ degrade to local and set `"degraded": "jina-unavailable: <reason>"`
  in the result (never silently switch). `local` / `jina` force a lane.
- Returns `{"success": true, "lane": ..., "count": n, "results": [{"message_id",
  "session_id", "role", "timestamp", "snippet", "distance"}...], "index": {"watermark",
  "pending_jina", "doc_count"}}` — a 0-count with a healthy index is an HONEST zero
  (success true), the index block making staleness visible.

Locking: `flock` (LOCK_EX, 20 s poll-timeout) around any collection open (index, backfill,
search all take it — single-owner store). Timeout ⇒ raise `StoreBusy`.

## Server (`zvec_memory_server.py`)

FastMCP, name `zvec-memory`, two tools:
- `memory_semantic_search(query: str, top_k: int = 5, lane: str = "auto") -> str` —
  flock → `index_catchup()` → `backfill_jina()` → `search()`; returns the JSON above.
  `StateDBUnavailable` ⇒ `{"success": false, "error": "state DB not found/unreadable: <path>"}`;
  `StoreBusy` ⇒ `{"success": false, "error": "memory store busy, retry"}`. Exceptions
  never escape as tracebacks.
- `memory_index_status() -> str` — watermark, doc_count, pending_jina, state_db path
  existence, lane availability (key file present?). No indexing, cheap.

## Tests (no network, no model download)

Fake embedders via monkeypatch (deterministic hash-based vectors, correct dims). Tmp
state.db built with the minimal `messages`/`sessions` DDL. Cases (red-on-demand — each
guards a failure that must be VISIBLE):
1. Missing state.db → explicit error (never empty success).
2. Lock held by another process → `StoreBusy` within timeout.
3. Watermark idempotency: second `index_catchup` indexes 0.
4. Honest zero: healthy index, no-match query → success true, count 0, index block present.
5. No key file → all docs pending; then fake-jina appears → `backfill_jina` drains queue.
6. Search results carry metadata + ascending distances; `lane` reflects reality.
7. Jina failure mid-search (fake raises) → result degraded to local with reason.
8. Only role user/assistant, content ≥12, active=1 rows are indexed.

## Pi deploy (README covers; deploy.sh automates the copy)

`~/zvec-memory/` (code + venv: `zvec fastembed mcp`), data at `~/.hermes/zvec-memory/`,
key file mode-600. Register: `claude mcp add -s user zvec-memory -- <venv-python>
<abs-path>/zvec_memory_server.py`. Allowlist in `~/.claude/settings.json`:
`mcp__zvec-memory__memory_semantic_search`, `mcp__zvec-memory__memory_index_status`.
Gateway restart. Live gates: raw JSON-RPC probe (initialize/tools-list/call), red-on-demand
(`ZVEC_MEMORY_STATE_DB=/nonexistent` → explicit error), then one real turn.
