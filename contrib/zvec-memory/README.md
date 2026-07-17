# zvec-memory — Pi runbook (brownet-coder)

Standalone stdio MCP server adding a **semantic search lane over conversation
history** (SPEC.md is the contract). Zero hermes-fork changes: the SDK loads
user-scope MCP servers. Read-only over `state.db`; owns a derived zvec index at
`~/.hermes/zvec-memory/` that is always rebuildable (delete the dir → next call
re-indexes from scratch).

Files: `zvec_memory_core.py` (library, no `mcp` dep) · `zvec_memory_server.py`
(FastMCP stdio) · `tests/test_zvec_memory.py` · `deploy.sh`.

## 1. Install (once, on the Pi)

```bash
ssh browbot@brownet-coder
mkdir -p ~/zvec-memory && cd ~/zvec-memory
python3 -m venv venv
venv/bin/pip install zvec fastembed mcp
```

Warm the local embedder ONCE while online (fastembed downloads the
bge-small-en-v1.5 ONNX model, ~130 MB, on first use — after this the local
lane needs no network):

```bash
venv/bin/python -c "import zvec_memory_core as c; print(len(c.embed_local(['warmup'])[0]))"
# expect: 384
```

## 2. Deploy the code (from the workstation)

```bash
bash deploy.sh        # scp + sha256 verify both sides; prints next steps
```

## 3. Env vars (all optional)

| Var | Default | Meaning |
|---|---|---|
| `ZVEC_MEMORY_DIR` | `~/.hermes/zvec-memory` | derived index home (`collection/`, `watermark.json`, `jina_pending.json`, `lock`) |
| `ZVEC_MEMORY_STATE_DB` | `~/.hermes/state.db` | Hermes state DB, opened strictly read-only |
| `ZVEC_MEMORY_JINA_KEY_FILE` | `<dir>/jina.key` | Jina API key file; **absent ⇒ local-only mode** (still fully functional) |

## 4. Jina key (optional quality lane)

```bash
install -m 600 /dev/null ~/.hermes/zvec-memory/jina.key
# paste the key (single line) into the file; verify:
stat -c '%a' ~/.hermes/zvec-memory/jina.key   # expect: 600
```

No key file = sovereignty lane only (`bge-small` local, ~13 ms/query on the
Pi). With the key, explicit searches ride jina-embeddings-v4 and degrade
LOUDLY to local when the API is unreachable (`"degraded"` field in the
result — never a silent switch).

## 5. Register the MCP server (user scope)

```bash
claude mcp add -s user zvec-memory -- \
  /home/browbot/zvec-memory/venv/bin/python \
  /home/browbot/zvec-memory/zvec_memory_server.py
```

## 6. Allowlist (headless denies un-allowlisted tools)

Add to the `permissions.allow` array in `~/.claude/settings.json`:

```json
"mcp__zvec-memory__memory_semantic_search",
"mcp__zvec-memory__memory_index_status"
```

## 7. Restart the gateway

Restart the Hermes gateway service so the SDK picks up the new user-scope
server, then confirm both tools appear in a fresh session.

## 8. Red-on-demand gates (make it go RED before trusting green)

**a. Raw JSON-RPC probe** (initialize → tools/list → one call), no agent
involved:

```bash
cd ~/zvec-memory
( printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"memory_index_status","arguments":{}}}' ; sleep 3 ) \
  | venv/bin/python zvec_memory_server.py
# expect: an initialize result, both tools listed, and a JSON status payload
```

**b. Missing state DB → explicit error, never an empty success:**

```bash
ZVEC_MEMORY_STATE_DB=/nonexistent/state.db \
  venv/bin/python -c "import zvec_memory_core as c; c.index_catchup()"
# expect: StateDBUnavailable naming /nonexistent/state.db
```

Through the server tool the same condition must answer
`{"success": false, "error": "state DB not found/unreadable: ..."}`.

**c. Busy store → StoreBusy, not a hang:** hold the lock in one shell…

```bash
venv/bin/python -c "
import fcntl, os, time
fd = os.open(os.path.expanduser('~/.hermes/zvec-memory/lock'), os.O_RDWR | os.O_CREAT)
fcntl.flock(fd, fcntl.LOCK_EX); print('holding 30s'); time.sleep(30)"
```

…and in another, expect `StoreBusy` after the ~20 s poll-timeout:

```bash
venv/bin/python -c "import zvec_memory_core as c; print(c.search('cache key'))"
```

**d. One real turn:** ask the agent something only history knows, e.g.
*"search memory: where did we fix the CI cache-key bug?"* — the reply must
carry `lane`, `count`, and the `index` block.

## 9. Operational notes

- The index is **derived**: `rm -rf ~/.hermes/zvec-memory` is always safe and
  rebuilds on the next tool call (watermark restarts at 0).
- `memory_index_status` is the cheap health check: watermark vs state.db,
  `pending_jina` (backfill queue), lane availability.
- Scores are cosine **distances** — ascending, smaller = closer.
- The zvec collection is single-owner; every open sits under an `flock`
  (`<dir>/lock`). Concurrent callers get "memory store busy, retry", not
  corruption.
