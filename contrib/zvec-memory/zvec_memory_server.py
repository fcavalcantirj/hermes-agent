"""zvec-memory MCP server: semantic search lane over Hermes conversation history.

FastMCP stdio server, name `zvec-memory`, two tools (SPEC §Server).
All logic lives in zvec_memory_core (no `mcp` import there); this file is the
thin transport. Tool errors come back as JSON `{"success": false, ...}` —
exceptions never escape as tracebacks.
"""

from __future__ import annotations

import json
import sys

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # explicit degrade: a missing dep must name itself
    sys.stderr.write(
        f"zvec-memory-server: the 'mcp' package is required "
        f"(pip install mcp into the server venv): {exc}\n"
    )
    raise SystemExit(1)

import zvec_memory_core as core

app = FastMCP("zvec-memory")


def _dump(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


@app.tool()
def memory_semantic_search(query: str, top_k: int = 5, lane: str = "auto") -> str:
    """Semantic search over indexed conversation history — finds past
    discussions by MEANING, not keywords.

    PREFER THIS over the keyword tool (session_search) whenever the question
    is phrased naturally: paraphrases, concepts, "where did we deal with…"
    questions that may not repeat the original words. Reach for
    session_search only when you know exact keywords/identifiers that
    appeared verbatim.

    Catches the index up from state.db, drains a slice of the jina backfill
    queue, then searches. lane: auto (jina when key present, degrade-loud to
    local) | local | jina. Returns JSON with results + an index-health block.
    """
    try:
        core.index_catchup()
        core.backfill_jina()
        return _dump(core.search(query, top_k=top_k, lane=lane))
    except core.StateDBUnavailable as exc:
        return _dump(
            {"success": False, "error": f"state DB not found/unreadable: {exc}"}
        )
    except core.StoreBusy:
        return _dump({"success": False, "error": "memory store busy, retry"})
    except Exception as exc:  # never a traceback over the wire
        return _dump({"success": False, "error": f"{type(exc).__name__}: {exc}"})


@app.tool()
def memory_index_status() -> str:
    """Index health: watermark, doc count, jina backfill queue, state.db
    presence, lane availability. No indexing, no embedding — cheap."""
    try:
        return _dump(core.index_status())
    except core.StoreBusy:
        return _dump({"success": False, "error": "memory store busy, retry"})
    except Exception as exc:
        return _dump({"success": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    app.run()  # FastMCP default transport is stdio
