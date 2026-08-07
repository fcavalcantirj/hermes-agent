"""Hermes-side stdio MCP proxy for remote `auth: oauth` mcp_servers entries.

The claude-agent-sdk fallback cannot hand its own OAuth tokens to the Claude
CLI it spawns — the CLI speaks stdio/http/sse MCP with no notion of Hermes's
token store. For a config.yaml `mcp_servers:` entry with `auth: oauth`
(Notion first), `_build_external_mcp_configs()` in
`claude_agent_sdk_session.py` instead points the CLI at THIS module as a
plain stdio server (`--server <name>`). This module authenticates to the
real remote using Hermes's OWN OAuth stack —
`tools.mcp_oauth_manager`'s provider over `HermesTokenStorage`, the single
token store — and pass-through proxies list_tools/call_tool over stdio, so
the SDK-spawned CLI needs zero OAuth awareness. Same subprocess pattern as
`agent/transports/hermes_tools_mcp_server.py`.

Posture is FAIL-SOFT after argv parsing: a missing/malformed config entry,
an OAuth build failure, a connect/initialize timeout — every one of them
logs to stderr and still serves a valid stdio MCP server with
`remote=None`, which the seams below turn into an empty tool catalog and
isError call results. A crashing child makes the CLI report a failed MCP
server (loud, and it can abort the session); an empty catalog degrades
quietly and matches the routing layer's own fail-soft contract. The SDK
CLI's handshake always completes.

Scope: tools only. Prompts, resources and sampling are not proxied, and
there is no reconnect machinery — this process lives exactly one SDK
session, so a remote that drops mid-session simply degrades (calls surface
as error results) instead of being nursed back.

Run with: python -m agent.transports.oauth_mcp_proxy --server <name>
Spawned by: _build_external_mcp_configs() in claude_agent_sdk_session.py
            for mcp_servers entries with `auth: oauth`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from io import TextIOWrapper

import mcp.types as types

logger = logging.getLogger(__name__)

# Slack on top of the entry's connect_timeout for the parts of the connect
# that timeout does NOT cover: provider build, dynamic client registration,
# token refresh, TLS. The inner bound stays on initialize() alone (#59349),
# this outer bound is the guarantee that a wedged OAuth refresh can never
# stall the CLI's handshake indefinitely.
# NOTE: the CLI applies its OWN startup timeout to this subprocess (MCP_TIMEOUT,
# 30s by default), so with a long connect_timeout the CLI may give up on the
# server before we finish connecting — raise MCP_TIMEOUT alongside it.
_CONNECT_SLACK_SECONDS = 15.0

# Matches the streamable-HTTP read timeout tools/mcp_tool.py uses: remote MCP
# servers routinely hold a response open for minutes on a slow tool call.
_REMOTE_READ_TIMEOUT = 300.0


def resolve_server_entry(name: str, mcp_config: dict) -> tuple[str, dict | None] | None:
    """Look up `name` in an already-loaded config.yaml `mcp_servers:` dict.

    Returns (url, oauth_cfg) when the entry is a dict with a truthy "url",
    oauth_cfg being entry.get("oauth") or None when absent. Returns None for
    a missing entry, a non-dict entry, or an entry without a url (e.g. a
    stdio `command`-based entry, which can never be an OAuth remote).
    """
    entry = mcp_config.get(name)
    if not isinstance(entry, dict):
        return None
    url = entry.get("url")
    if not url:
        return None
    return (url, entry.get("oauth") or None)


def build_remote_auth(name: str, url: str, oauth_cfg: dict | None):
    """Build the httpx auth for the remote OAuth MCP server.

    Lazy import: tools.mcp_oauth_manager is the ONLY place allowed to
    instantiate the SDK's OAuthClientProvider (shared HermesTokenStorage,
    disk-watch reload, 401 dedup) — never roll OAuth here.
    """
    from tools.mcp_oauth_manager import get_manager

    return get_manager().get_or_build_provider(name, url, oauth_cfg)


async def proxy_list_tools(remote) -> list:
    """Pass-through tools/list, walking pagination.

    Fail-soft floor: when `remote` is None (startup connect failed — tokens
    revoked between routing and spawn, remote down, etc.), serve an empty
    catalog instead of raising, so the SDK CLI's handshake completes
    cleanly.
    """
    if remote is None:
        return []

    tools: list = []
    result = await remote.list_tools(cursor=None)
    tools.extend(result.tools)
    while result.nextCursor:
        result = await remote.list_tools(cursor=result.nextCursor)
        tools.extend(result.tools)
    return tools


async def proxy_call_tool(remote, name: str, arguments) -> types.CallToolResult:
    """Pass-through tools/call.

    When `remote` is None, return an explicit error CallToolResult (never
    raise) explaining the upstream OAuth MCP server is unavailable because
    the startup connection failed. Otherwise return the remote's
    CallToolResult UNCHANGED — never re-wrapped or re-serialized.
    """
    if remote is None:
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=(
                        "Upstream OAuth MCP server is unavailable: the "
                        "startup connection failed."
                    ),
                )
            ],
            isError=True,
        )
    return await remote.call_tool(name, arguments)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _resolve_target(name: str) -> tuple[str, dict | None, dict, float] | None:
    """(url, oauth_cfg, headers, connect_timeout) for `name`, or None.

    None means "serve the empty catalog": there is no such entry, it is not a
    remote, or config could not be read at all. Never raises — this runs
    before the stdio handshake and a raise here would kill the child.

    `resolve_server_entry` owns the is-this-a-remote decision; the extra
    transport knobs (headers, connect_timeout) are read off the same entry
    dict, which that seam has already proven to be a dict with a url.
    """
    try:
        from tools.mcp_tool import _DEFAULT_CONNECT_TIMEOUT, _load_mcp_config

        mcp_config = _load_mcp_config() or {}
    except Exception as exc:
        logger.warning("oauth proxy '%s': cannot read mcp_servers config: %s", name, exc)
        return None

    resolved = resolve_server_entry(name, mcp_config)
    if resolved is None:
        logger.warning(
            "oauth proxy '%s': no remote mcp_servers entry with a url in "
            "config.yaml — serving an empty tool catalog",
            name,
        )
        return None
    url, oauth_cfg = resolved

    entry = mcp_config.get(name) or {}
    raw_headers = entry.get("headers")
    headers = (
        {str(k): str(v) for k, v in raw_headers.items()}
        if isinstance(raw_headers, dict)
        else {}
    )
    # ssl_verify / client certs are deliberately NOT carried over: a remote
    # needing them fails the connect and degrades to the empty catalog rather
    # than being served over a weaker-than-configured transport.
    try:
        connect_timeout = float(entry.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT))
    except (TypeError, ValueError):
        connect_timeout = float(_DEFAULT_CONNECT_TIMEOUT)
    return (url, oauth_cfg, headers, max(connect_timeout, 1.0))


# ---------------------------------------------------------------------------
# Remote connect + stdio serve lifecycle
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _connect_remote(
    name: str,
    url: str,
    oauth_cfg: dict | None,
    headers: dict,
    connect_timeout: float,
):
    """Yield an initialized `mcp.ClientSession` against the remote.

    Same streamable-HTTP stack as `tools.mcp_tool._run_http`'s new-API branch:
    a caller-owned httpx client (the SDK skips cleanup when `http_client` is
    passed) carrying the OAuth auth, and a handshake bounded by
    `connect_timeout` — an endpoint that accepts the connection but never
    answers `initialize` would otherwise park this coroutine forever (#59349).
    The contexts stay open for as long as the caller holds the yield, i.e. the
    whole stdio serve loop.

    BOTH timeouts live in here, and the caller must NOT wrap the enter in
    `asyncio.wait_for`: wait_for runs the awaitable in a fresh Task, so the
    anyio cancel scopes below would be entered in that task and exited in the
    caller's at teardown ("Attempted to exit cancel scope in a different
    task"). The outer bound is an anyio cancel scope on the CURRENT task that
    wraps the transport contexts and is DISARMED (deadline → inf) the moment
    the handshake lands: it cannot close before the transport it encloses
    (cancel scopes unwind strictly LIFO) and it must not stay armed over the
    serve loop, so disarm-in-place is the only shape that gives both.
    """
    import math

    import anyio
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    # Via tools.mcp_tool so its fallback for SDK builds that don't export the
    # constant applies here too.
    from tools.mcp_tool import LATEST_PROTOCOL_VERSION

    auth = build_remote_auth(name, url, oauth_cfg)
    if auth is None:
        # SDK auth module unavailable. Try anyway: a 401 degrades to the
        # empty catalog through the caller's fail-soft path, and a remote that
        # does not actually enforce auth still works.
        logger.warning(
            "oauth proxy '%s': OAuth provider unavailable — connecting unauthenticated",
            name,
        )

    request_headers = dict(headers)
    # Some MCP servers reject session-less POSTs without this on the initial
    # initialize; user-supplied casing wins (mirrors _run_http).
    if not any(key.lower() == "mcp-protocol-version" for key in request_headers):
        request_headers["mcp-protocol-version"] = LATEST_PROTOCOL_VERSION

    client_kwargs: dict = {
        "follow_redirects": True,
        "timeout": httpx.Timeout(float(connect_timeout), read=_REMOTE_READ_TIMEOUT),
        "headers": request_headers,
    }
    if auth is not None:
        client_kwargs["auth"] = auth

    # Outer bound over the whole attempt (transport setup + handshake + any
    # token refresh the auth flow triggers): a wedged OAuth refresh must never
    # stall the CLI's own handshake, which waits on ours.
    with anyio.move_on_after(
        float(connect_timeout) + _CONNECT_SLACK_SECONDS
    ) as connect_scope:
        async with httpx.AsyncClient(**client_kwargs) as http_client:
            async with streamable_http_client(url, http_client=http_client) as (
                read_stream,
                write_stream,
                _get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    # Inner bound on initialize alone (#59349).
                    await asyncio.wait_for(
                        session.initialize(), timeout=float(connect_timeout)
                    )
                    connect_scope.deadline = math.inf  # connected: disarm
                    logger.info(
                        "oauth proxy '%s': remote session initialized", name
                    )
                    yield session
                    return
    raise TimeoutError(
        f"remote connect exceeded "
        f"{float(connect_timeout) + _CONNECT_SLACK_SECONDS:.0f}s"
    )


async def _run_stdio(name: str, remote, stdout_buffer=None) -> None:
    """Serve the stdio MCP wire, delegating both handlers to the seams.

    Called exactly once per process, with a live `remote` or with None — the
    single call site is why the fail-soft path cannot drift from the happy one.
    """
    import anyio
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    server = Server(name)

    @server.list_tools()
    async def _list_tools() -> list:
        return await proxy_list_tools(remote)

    # validate_input=False: a pass-through pipe must not re-validate arguments
    # against schemas it does not own — the remote is the authority, and a
    # local validator can only invent rejections the remote would have
    # accepted. Exceptions from the seam are turned into error CallToolResults
    # by the lowlevel Server, so a remote that drops mid-session surfaces as an
    # error result with its own message and needs no second wrapping layer here.
    @server.call_tool(validate_input=False)
    async def _call_tool(tool_name: str, arguments) -> types.CallToolResult:
        return await proxy_call_tool(remote, tool_name, arguments)

    # The wire is an explicit handle rather than sys.stdout, because main()
    # points sys.stdout at stderr for the whole run (Hermes' config/env
    # loaders print freely, and one stray line breaks the CLI's JSON parse).
    buffer = stdout_buffer if stdout_buffer is not None else sys.stdout.buffer
    wire = anyio.wrap_file(TextIOWrapper(buffer, encoding="utf-8"))
    async with stdio_server(stdout=wire) as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


async def _serve(name: str, target, stdout_buffer=None) -> None:
    """Connect the remote (best effort), then serve stdio for its lifetime.

    Connect-then-serve, not serve-then-connect: the CLI's `initialize` is
    answered only after the remote is up (or has failed), so the tool catalog
    it caches at startup is the real one instead of an empty list that a later
    connect could never correct.
    """
    async with contextlib.AsyncExitStack() as stack:
        remote = None
        if target is not None:
            url, oauth_cfg, headers, connect_timeout = target
            # Headless child: stdin is the MCP wire, not a TTY. An expired
            # token whose refresh fails must degrade to the empty-catalog path,
            # never park on a browser/paste flow. Held for the WHOLE serve, not
            # just the connect — a mid-session refresh runs in this same
            # context (the suppression is a ContextVar, inherited by the
            # per-request tasks the server spawns from here).
            try:
                from tools.mcp_oauth import suppress_interactive_oauth

                stack.enter_context(suppress_interactive_oauth())
            except Exception as exc:
                logger.debug("oauth proxy '%s': interactive suppression unavailable: %s", name, exc)
            try:
                # Entered on THIS task and closed by this stack on the way out
                # — no wait_for wrapper (see _connect_remote's docstring: its
                # timeouts are internal precisely so the anyio cancel scopes
                # are entered and exited in the same task).
                remote = await stack.enter_async_context(
                    _connect_remote(name, url, oauth_cfg, headers, connect_timeout)
                )
            except BaseExceptionGroup as eg:
                # anyio transport TaskGroup failure. Re-raise a group that is
                # purely cancellation — that is the process being torn down,
                # not a remote we should paper over.
                _cancelled, rest = eg.split(asyncio.CancelledError)
                if rest is None:
                    raise
                logger.warning(
                    "oauth proxy '%s': remote connect failed (%r) — serving an "
                    "empty tool catalog", name, rest,
                )
                remote = None
            except Exception as exc:
                # OAuthNonInteractiveError, TimeoutError, HTTP 401/403/5xx,
                # DNS, malformed initialize — all the same degrade.
                logger.warning(
                    "oauth proxy '%s': remote connect failed (%s: %s) — serving "
                    "an empty tool catalog", name, type(exc).__name__, exc,
                )
                remote = None
        await _run_stdio(name, remote, stdout_buffer)


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m agent.transports.oauth_mcp_proxy`."""
    parser = argparse.ArgumentParser(
        prog="python -m agent.transports.oauth_mcp_proxy",
        description=(
            "Hermes-side stdio MCP proxy for a remote `auth: oauth` "
            "mcp_servers entry."
        ),
    )
    parser.add_argument(
        "--server", required=True, help="config.yaml mcp_servers key to proxy"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="log at INFO instead of WARNING"
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        stream=sys.stderr,  # MCP uses stdio for protocol — logs MUST go to stderr
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")

    # Grab the real wire before sys.stdout is pointed at stderr for the rest of
    # the run: config loading, dotenv and OAuth helpers all print, and quiet
    # mode is not a guarantee. _run_stdio writes the protocol to this handle.
    sys.stdout.flush()
    stdout_buffer = sys.stdout.buffer

    with contextlib.redirect_stdout(sys.stderr):
        # Credentials channel (C4-compliant): read ~/.hermes/.env from DISK
        # inside this child, like every other Hermes entry point. The spawn env
        # is a minimal non-secret allowlist, so without this load the OAuth
        # stack cannot see .env-stored client ids/secrets.
        try:
            from hermes_cli.env_loader import load_hermes_dotenv

            load_hermes_dotenv()
        except Exception:
            logger.debug("hermes dotenv load failed", exc_info=True)

        target = _resolve_target(args.server)
        try:
            asyncio.run(_serve(args.server, target, stdout_buffer))
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            # Post-handshake failures only (everything before serving is
            # fail-soft): the wire is already dead, so report it.
            logger.exception("oauth proxy '%s' crashed", args.server)
            sys.stderr.write(f"oauth proxy '{args.server}' error: {exc}\n")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
