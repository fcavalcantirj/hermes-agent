"""Hermes-tools-as-MCP server for the codex_app_server runtime.

When the user runs `openai/*` turns through the codex app-server, codex
owns the loop and builds its own tool list. By default, that means
Hermes' richer tool surface — web search, browser automation,
delegate_task subagents, vision analysis, persistent memory, skills,
cross-session search, image generation, TTS — is unreachable.

This module exposes a curated subset of those Hermes tools to the
spawned codex subprocess via stdio MCP. Codex registers it as a normal
MCP server (per `~/.codex/config.toml [mcp_servers.hermes-tools]`) and
the user gets full Hermes capability inside a Codex turn.

Scope (what we expose):
  - web_search, web_extract              — Firecrawl, no codex equivalent
  - browser_navigate / _click / _type /  — Camofox/Browserbase automation
    _snapshot / _scroll / _back / _press /
    _get_images / _console / _vision
  - vision_analyze                       — image inspection by vision model
  - image_generate                       — image generation
  - skill_view, skills_list              — Hermes' skill library
  - text_to_speech                       — TTS
  - kanban_* (complete/block/comment/    — kanban worker + orchestrator
    heartbeat/show/list/create/            handoff (stateless: read env var,
    unblock/link)                          write ~/.hermes/kanban.db)

What we DO NOT expose:
  - terminal / shell                     — codex's own shell tool
  - read_file / write_file / patch       — codex's apply_patch + shell
  - search_files / process               — codex's shell
  - clarify                              — codex's own UX
  - delegate_task / todo                 — `_AGENT_LOOP_TOOLS` in Hermes
                                           (model_tools.py). They require
                                           the running AIAgent context to
                                           dispatch (mid-loop state), so a
                                           stateless MCP callback can't
                                           drive them. See the inline
                                           comment on EXPOSED_TOOLS below.

Exposed via STATELESS SHIMS (#26567) rather than the generic dispatcher:
  - memory                               — tools.memory_tool.load_on_disk_store()
                                           per call: native caps, drift guard,
                                           threat scan, locking all inherited
  - session_search                       — read-only SessionDB over the state
                                           DB; the calling session's id rides
                                           HERMES_MCP_SESSION_ID
  The `_AGENT_LOOP_TOOLS` refusal in handle_function_call stays intact for
  every other caller — the shims are dedicated closures, not a widened gate.

Run with: python -m agent.transports.hermes_tools_mcp_server
Spawned by: CodexAppServerSession.ensure_started() when the runtime is
            active and config opts in.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Tools we expose. Each name MUST match a registered Hermes tool that
# `model_tools.handle_function_call()` can dispatch.
#
# What we deliberately DO NOT expose:
#   - terminal / shell / read_file / write_file / patch / search_files /
#     process — codex's built-ins cover these and approval routes through
#     codex's own UI.
#   - delegate_task / memory / session_search / todo — these are
#     `_AGENT_LOOP_TOOLS` in Hermes (model_tools.py:493). They require
#     the running AIAgent context to dispatch (mid-loop state), so a
#     stateless MCP callback can't drive them. Hermes' default runtime
#     keeps these working; the codex_app_server runtime cannot.
EXPOSED_TOOLS: tuple[str, ...] = (
    "web_search",
    "web_extract",
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_snapshot",
    "browser_scroll",
    "browser_back",
    "browser_get_images",
    "browser_console",
    "browser_vision",
    "vision_analyze",
    "image_generate",
    "skill_view",
    "skills_list",
    "text_to_speech",
    # Kanban worker handoff tools — gated on HERMES_KANBAN_TASK env var
    # (set by the kanban dispatcher when spawning a worker). Without these
    # in the callback, a worker spawned with openai_runtime=codex_app_server
    # could do the work but couldn't report completion back to the kernel,
    # making it hang until timeout. Stateless dispatch — they just read
    # the env var and write to ~/.hermes/kanban.db.
    "kanban_complete",
    "kanban_block",
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_show",
    "kanban_list",
    # NOTE: kanban_create / kanban_unblock / kanban_link are orchestrator-
    # only — the kanban tool gates them on HERMES_KANBAN_TASK being unset.
    # They're exposed here for orchestrator agents running on the codex
    # runtime that need to dispatch new tasks.
    "kanban_create",
    "kanban_unblock",
    "kanban_link",
)


# --- Stateless agent-loop shims (#26567) ---------------------------------
#
# `memory` and `session_search` are `_AGENT_LOOP_TOOLS`: the generic
# dispatcher refuses them because natively they receive live agent state
# (the loop's MemoryStore / session-DB handle) from tool_executor. Both
# have faithful stateless equivalents, so runtimes that own their own agent
# loop (claude-agent-sdk, codex app-server) can regain them through this
# server without touching that refusal:
#
#   memory         → a fresh `load_on_disk_store()` per call. Char caps,
#                    config overrides, external-drift guard, threat scan,
#                    file locking and the consolidation-failure breaker all
#                    live in MemoryStore/memory_tool and are inherited.
#                    NOTE: shim writes do not mirror to an external
#                    MemoryProvider — that hook lives in the native tool
#                    executor. Acceptable for provider-less deployments.
#   session_search → `SessionDB(read_only=True)` over the state DB (never a
#                    writable handle in a model-facing subprocess). The
#                    calling session's id arrives via HERMES_MCP_SESSION_ID
#                    so own-lineage exclusion keeps working; the DB path can
#                    be overridden via HERMES_MCP_STATE_DB (defaults to the
#                    profile's state.db).

_SESSION_ID_ENV = "HERMES_MCP_SESSION_ID"
_STATE_DB_ENV = "HERMES_MCP_STATE_DB"


def dispatch_memory(kwargs: dict) -> str:
    """Stateless `memory` dispatch: native handler + on-disk store."""
    from tools.memory_tool import load_on_disk_store, memory_tool

    return memory_tool(
        action=kwargs.get("action", ""),
        target=kwargs.get("target", "memory"),
        content=kwargs.get("content"),
        old_text=kwargs.get("old_text"),
        operations=kwargs.get("operations"),
        store=load_on_disk_store(),
    )


def dispatch_session_search(kwargs: dict) -> str:
    """Stateless `session_search` dispatch: read-only DB + env session id."""
    from pathlib import Path

    import hermes_state
    from tools import session_search_tool

    db_path = Path(
        os.environ.get(_STATE_DB_ENV, "").strip()
        or hermes_state.DEFAULT_DB_PATH
    )
    if not db_path.exists():
        # Explicit degrade — a missing DB must never read as "no results".
        return json.dumps({
            "success": False,
            "error": f"session_search unavailable: state DB not found at {db_path}",
        })
    try:
        db = hermes_state.SessionDB(db_path=db_path, read_only=True)
    except Exception as exc:
        return json.dumps({
            "success": False,
            "error": f"session_search unavailable: cannot open state DB read-only: {exc}",
        })
    try:
        return session_search_tool.session_search(
            query=kwargs.get("query") or "",
            role_filter=kwargs.get("role_filter"),
            limit=kwargs.get("limit", 3),
            session_id=kwargs.get("session_id"),
            around_message_id=kwargs.get("around_message_id"),
            window=kwargs.get("window", 5),
            sort=kwargs.get("sort"),
            profile=kwargs.get("profile"),
            db=db,
            current_session_id=os.environ.get(_SESSION_ID_ENV, "").strip() or None,
        )
    finally:
        try:
            db.close()
        except Exception:
            pass


def _memory_enabled_in_config() -> bool:
    """Honor the operator's `memory.memory_enabled` config (default on)."""
    try:
        from hermes_cli.config import load_config

        return bool(
            ((load_config() or {}).get("memory", {}) or {}).get("memory_enabled", True)
        )
    except Exception:
        return True


def _stateless_shim_defs() -> list:
    """(name, description, handler) triples to register, honoring config.

    session_search is always defined — a missing state DB degrades to an
    explicit error at call time, which is more diagnosable than an absent
    tool. memory respects the config kill-switch.
    """
    defs = []
    if _memory_enabled_in_config():
        from tools.memory_tool import MEMORY_SCHEMA

        defs.append(("memory", MEMORY_SCHEMA.get("description", "Hermes memory tool"),
                     dispatch_memory))
    from tools.session_search_tool import SESSION_SEARCH_SCHEMA

    defs.append((
        "session_search",
        SESSION_SEARCH_SCHEMA.get("description", "Search past Hermes sessions"),
        dispatch_session_search,
    ))
    return defs


def _build_server() -> Any:
    """Create the FastMCP server with Hermes tools attached. Lazy imports
    so the module can be imported without the mcp package installed
    (we degrade to a clear error only when actually run)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - install hint
        raise ImportError(
            f"hermes-tools MCP server requires the 'mcp' package: {exc}"
        ) from exc

    # Discover Hermes tools so dispatch works.
    from model_tools import (
        get_tool_definitions,
        handle_function_call,
    )

    mcp = FastMCP(
        "hermes-tools",
        instructions=(
            "Hermes Agent's tool surface, exposed for use inside a Codex "
            "session. Use these for capabilities Codex's built-in toolset "
            "doesn't cover: web search/extract, browser automation, "
            "subagent delegation, vision, image generation, persistent "
            "memory, skills, and cross-session search."
        ),
    )

    # Pull authoritative Hermes tool schemas for the ones we expose, so
    # MCP clients see the same parameter docs Hermes gives the model.
    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }

    exposed_count = 0

    for name in EXPOSED_TOOLS:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug(
                "skipping %s — not registered in this Hermes process", name
            )
            continue

        description = spec.get("description") or f"Hermes {name} tool"
        params_schema = spec.get("parameters") or {"type": "object", "properties": {}}

        # FastMCP wants a Python callable. Build a closure that takes the
        # arguments dict, dispatches via handle_function_call, and returns
        # the result string. We use add_tool() for full control over the
        # input schema (FastMCP's @tool() decorator inspects type hints,
        # which we can't get from a JSON schema at runtime).
        def _make_handler(tool_name: str):
            def _dispatch(**kwargs: Any) -> str:
                try:
                    return handle_function_call(tool_name, kwargs or {})
                except Exception as exc:
                    logger.exception("tool %s raised", tool_name)
                    return json.dumps({"error": str(exc), "tool": tool_name})
            _dispatch.__name__ = tool_name
            _dispatch.__doc__ = description
            return _dispatch

        try:
            mcp.add_tool(
                _make_handler(name),
                name=name,
                description=description,
                # FastMCP accepts JSON schema directly via the
                # input_schema parameter on newer versions; older
                # versions use parameters_schema. Try both for compat.
            )
        except TypeError:
            # Older mcp SDK signature — fall back to decorator-style.
            handler = _make_handler(name)
            handler = mcp.tool(name=name, description=description)(handler)

        exposed_count += 1

    # Stateless agent-loop shims (#26567) — registered as dedicated
    # closures so handle_function_call's `_AGENT_LOOP_TOOLS` refusal stays
    # intact for every other caller.
    shim_count = 0
    for shim_name, shim_description, shim_fn in _stateless_shim_defs():

        def _make_shim_handler(fn, tool_name: str, description: str):
            def _dispatch(**kwargs: Any) -> str:
                try:
                    return fn(kwargs or {})
                except Exception as exc:
                    logger.exception("shim tool %s raised", tool_name)
                    return json.dumps({"error": str(exc), "tool": tool_name})
            _dispatch.__name__ = tool_name
            _dispatch.__doc__ = description
            return _dispatch

        shim_handler = _make_shim_handler(shim_fn, shim_name, shim_description)
        try:
            mcp.add_tool(shim_handler, name=shim_name, description=shim_description)
        except TypeError:
            shim_handler = mcp.tool(name=shim_name, description=shim_description)(
                shim_handler
            )
        shim_count += 1

    logger.info(
        "hermes-tools MCP server registered %d/%d tools + %d stateless shims",
        exposed_count,
        len(EXPOSED_TOOLS),
        shim_count,
    )
    return mcp


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for `python -m agent.transports.hermes_tools_mcp_server`."""
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv

    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,  # MCP uses stdio for protocol — logs MUST go to stderr
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Quiet mode: keep Hermes' own banners off stdout (which is the MCP wire).
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")

    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write(f"hermes-tools MCP server cannot start: {exc}\n")
        return 2

    # FastMCP runs with stdio transport by default when launched as a
    # subprocess.
    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("hermes-tools MCP server crashed")
        sys.stderr.write(f"hermes-tools MCP server error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
