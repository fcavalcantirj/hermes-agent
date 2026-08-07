"""RED tests for agent.transports.oauth_mcp_proxy.

The proxy is the Hermes-side stdio MCP server the claude-agent-sdk fallback
spawns for remote `auth: oauth` servers (Notion first). It authenticates to
the real remote with Hermes's OWN OAuth stack — tools.mcp_oauth_manager's
provider over HermesTokenStorage, the single token store — and pass-through
proxies list_tools/call_tool over stdio, so the SDK-spawned CLI needs zero
OAuth awareness (the exact hermes_tools_mcp_server pattern).

New file rather than test_claude_sdk_runtime.py on purpose: nothing here
builds a ClaudeAgentSdkSession — these are unit tests for the proxy module's
own seams (config-entry resolution, auth construction, pass-through
handlers), and the runtime file is already ~1600 lines of session/projector
coverage. The ROUTING into the proxy (stdio-vs-http-vs-sse decision) is
locked in test_claude_sdk_runtime.py::TestOAuthProxyRouting, next to the
merge code it tests.

No test here touches the network or real credentials: the remote MCP session
is a duck-typed fake and the OAuth manager is stubbed.
"""

import asyncio
import importlib

import mcp.types as types
import pytest


def _proxy():
    """Import inside the test body: pre-implementation every test fails
    with a clear ModuleNotFoundError (RED on the missing module), instead
    of one opaque collection error hiding the rest of the file."""
    return importlib.import_module("agent.transports.oauth_mcp_proxy")


# ---------- config-entry resolution (pure decision seam) ----------


class TestResolveServerEntry:
    """resolve_server_entry(name, mcp_config) -> (url, oauth_cfg) | None.

    Pure function over the already-loaded config.yaml `mcp_servers:` dict —
    the proxy child re-reads config itself (argv carries only --server NAME),
    so this seam is testable without subprocess, network, or token files.
    """

    def test_well_formed_oauth_entry_resolves(self):
        cfg = {
            "notion": {
                "url": "https://mcp.notion.com/mcp",
                "auth": "oauth",
                "oauth": {"scope": "read"},
            },
        }
        assert _proxy().resolve_server_entry("notion", cfg) == (
            "https://mcp.notion.com/mcp",
            {"scope": "read"},
        )

    def test_missing_oauth_block_resolves_with_none_cfg(self):
        # The `oauth:` block is optional in config.yaml (defaults applied
        # downstream by apply_oauth_provider_defaults) — its absence must
        # not read as "malformed".
        cfg = {"notion": {"url": "https://mcp.notion.com/mcp", "auth": "oauth"}}
        assert _proxy().resolve_server_entry("notion", cfg) == (
            "https://mcp.notion.com/mcp",
            None,
        )

    def test_unknown_server_returns_none(self):
        assert _proxy().resolve_server_entry("ghost", {}) is None

    def test_entry_without_url_returns_none(self):
        # A stdio entry (command, no url) can never be an OAuth remote.
        cfg = {"local": {"command": "npx", "args": ["-y", "srv"]}}
        assert _proxy().resolve_server_entry("local", cfg) is None

    def test_non_dict_entry_returns_none(self):
        assert _proxy().resolve_server_entry("s", {"s": "https://x/mcp"}) is None


# ---------- OAuth auth construction (reuse, don't reimplement) ----------


class TestBuildRemoteAuth:
    def test_delegates_to_central_oauth_manager(self, monkeypatch):
        # The proxy must obtain its httpx auth from
        # tools.mcp_oauth_manager.get_manager().get_or_build_provider(...)
        # — the ONE place allowed to instantiate OAuthClientProvider
        # (shared HermesTokenStorage, disk-watch reload, 401 dedup) — never
        # roll its own OAuth. Patching the SOURCE module also pins the
        # implementation to a lazy in-function import, matching repo
        # convention for heavy tool modules.
        import tools.mcp_oauth_manager as mgr_mod

        sentinel = object()
        calls = {}

        class _FakeManager:
            def get_or_build_provider(self, name, url, oauth_cfg):
                calls["args"] = (name, url, oauth_cfg)
                return sentinel

        monkeypatch.setattr(mgr_mod, "get_manager", lambda: _FakeManager())
        auth = _proxy().build_remote_auth(
            "notion", "https://mcp.notion.com/mcp", {"scope": "read"}
        )
        assert auth is sentinel
        assert calls["args"] == (
            "notion",
            "https://mcp.notion.com/mcp",
            {"scope": "read"},
        )


# ---------- pass-through handlers (fake remote, no network) ----------


def _tool(name):
    return types.Tool(name=name, inputSchema={"type": "object", "properties": {}})


class _FakeRemote:
    """Duck-typed mcp.ClientSession: scripted list_tools pages, recorded
    call_tool invocations."""

    def __init__(self, pages=None, call_result=None):
        self.pages = list(pages or [])
        self.call_result = call_result
        self.cursors = []
        self.calls = []

    async def list_tools(self, cursor=None, **kwargs):
        self.cursors.append(cursor)
        return self.pages.pop(0)

    async def call_tool(self, name, arguments=None, **kwargs):
        self.calls.append((name, arguments))
        return self.call_result


class TestProxyListTools:
    def test_single_page_passes_tools_through(self):
        remote = _FakeRemote(
            pages=[types.ListToolsResult(tools=[_tool("notion-search")])]
        )
        tools = asyncio.run(_proxy().proxy_list_tools(remote))
        assert [t.name for t in tools] == ["notion-search"]

    def test_paginates_until_cursor_exhausted(self):
        # Real remotes (Notion serves 25+ tools) may page tools/list; the
        # proxy must walk nextCursor, not silently serve page one.
        remote = _FakeRemote(
            pages=[
                types.ListToolsResult(tools=[_tool("a")], nextCursor="c2"),
                types.ListToolsResult(tools=[_tool("b")]),
            ]
        )
        tools = asyncio.run(_proxy().proxy_list_tools(remote))
        assert [t.name for t in tools] == ["a", "b"]
        assert remote.cursors == [None, "c2"]

    def test_no_remote_serves_zero_tools(self):
        # Fail-soft floor: when the startup connect failed (tokens revoked
        # between routing and spawn, remote down), the proxy still answers
        # tools/list — with an empty catalog — so the SDK CLI's handshake
        # completes cleanly instead of the server crashing the session.
        assert asyncio.run(_proxy().proxy_list_tools(None)) == []


class TestProxyCallTool:
    def test_result_passes_through_unchanged(self):
        result = types.CallToolResult(
            content=[types.TextContent(type="text", text="hit")]
        )
        remote = _FakeRemote(call_result=result)
        out = asyncio.run(
            _proxy().proxy_call_tool(remote, "notion-search", {"query": "q"})
        )
        # Thin pipe: the remote's CallToolResult is returned as-is (the
        # lowlevel Server accepts CallToolResult from a call_tool handler),
        # never re-wrapped or re-serialized.
        assert out is result
        assert remote.calls == [("notion-search", {"query": "q"})]

    def test_error_result_stays_error(self):
        result = types.CallToolResult(
            content=[types.TextContent(type="text", text="401 unauthorized")],
            isError=True,
        )
        remote = _FakeRemote(call_result=result)
        out = asyncio.run(_proxy().proxy_call_tool(remote, "t", {}))
        assert out.isError is True

    def test_no_remote_returns_error_result_not_exception(self):
        out = asyncio.run(_proxy().proxy_call_tool(None, "t", {}))
        assert isinstance(out, types.CallToolResult)
        assert out.isError is True
        assert out.content  # says WHY, never an empty shrug
