"""Tests for the stateless memory/session_search shims in the hermes-tools
MCP server (#26567).

Natively `memory` and `session_search` are `_AGENT_LOOP_TOOLS`: the generic
dispatcher refuses them because they need live AIAgent state. The shims
supply that state statelessly — `load_on_disk_store()` per call for memory,
a read-only `SessionDB` + the calling session's id from env for
session_search — so an agent whose loop is owned by an external runtime
(claude-agent-sdk, codex app-server) regains both tools.

No `mcp` package required: the dispatch functions are plain module-level
callables; only `_build_server()` (not under test here) needs FastMCP.

Plant-the-failure discipline: the DB-missing path must yield an EXPLICIT
error (never a silently-empty result), and the refusal in
`handle_function_call` must remain intact for non-shim callers.
"""

import json

import pytest

from agent.transports.hermes_tools_mcp_server import (
    _stateless_shim_defs,
    dispatch_memory,
    dispatch_session_search,
)


@pytest.fixture()
def tmp_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MCP_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_MCP_STATE_DB", raising=False)
    return home


class TestMemoryShim:
    def test_write_lands_in_canonical_memories_dir(self, tmp_hermes_home):
        out = json.loads(
            dispatch_memory(
                {"action": "add", "target": "memory", "content": "auth refactor merged to main"}
            )
        )
        assert out.get("success") is True
        memory_file = tmp_hermes_home / "memories" / "MEMORY.md"
        assert memory_file.exists()
        assert "auth refactor merged to main" in memory_file.read_text()

    def test_native_caps_enforced(self, tmp_hermes_home):
        # The shim reuses the native store: an oversized add must be rejected
        # with the native consolidation error, not silently truncated.
        out = json.loads(
            dispatch_memory({"action": "add", "target": "memory", "content": "x" * 5000})
        )
        assert out.get("success") is False
        assert "exceed" in json.dumps(out).lower()

    def test_batch_operations_supported(self, tmp_hermes_home):
        out = json.loads(
            dispatch_memory(
                {
                    "target": "memory",
                    "operations": [
                        {"action": "add", "content": "fact alpha"},
                        {"action": "add", "content": "fact beta"},
                    ],
                }
            )
        )
        assert out.get("success") is True
        content = (tmp_hermes_home / "memories" / "MEMORY.md").read_text()
        assert "fact alpha" in content and "fact beta" in content


class TestSessionSearchShim:
    def _seed_db(self, path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=path)
        db.create_session("sess-hist-1", source="telegram")
        db.append_message("sess-hist-1", "user", "when did we merge the auth refactor?")
        db.append_message("sess-hist-1", "assistant", "The auth refactor merged on Thursday.")
        db.close()

    def test_search_returns_seeded_rows(self, tmp_hermes_home, monkeypatch):
        db_path = tmp_hermes_home / "state.db"
        self._seed_db(db_path)
        monkeypatch.setenv("HERMES_MCP_STATE_DB", str(db_path))
        out = dispatch_session_search({"query": "auth refactor"})
        assert "auth refactor" in out
        assert "sess-hist-1" in out

    def test_missing_db_yields_explicit_error(self, tmp_hermes_home, monkeypatch):
        # RED-first: a missing state DB must surface as an explicit error,
        # never as a silently-empty result set.
        monkeypatch.setenv(
            "HERMES_MCP_STATE_DB", str(tmp_hermes_home / "nope" / "state.db")
        )
        out = json.loads(dispatch_session_search({"query": "anything"}))
        assert out.get("success") is False
        assert "state DB" in out.get("error", "")

    def test_current_session_id_rides_env(self, tmp_hermes_home, monkeypatch):
        db_path = tmp_hermes_home / "state.db"
        self._seed_db(db_path)
        monkeypatch.setenv("HERMES_MCP_STATE_DB", str(db_path))
        monkeypatch.setenv("HERMES_MCP_SESSION_ID", "sess-current-9")

        captured = {}
        import tools.session_search_tool as sst

        real = sst.session_search

        def spy(**kwargs):
            captured.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(sst, "session_search", spy)
        dispatch_session_search({"query": "auth"})
        assert captured.get("current_session_id") == "sess-current-9"

    def test_db_opened_read_only(self, tmp_hermes_home, monkeypatch):
        # The shim must never hand a writable DB handle to a model-facing
        # subprocess. SessionDB(read_only=True) attaches with mode=ro.
        db_path = tmp_hermes_home / "state.db"
        self._seed_db(db_path)
        monkeypatch.setenv("HERMES_MCP_STATE_DB", str(db_path))

        captured = {}
        import agent.transports.hermes_tools_mcp_server as srv
        import hermes_state

        real = hermes_state.SessionDB

        class SpyDB(real):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(hermes_state, "SessionDB", SpyDB)
        dispatch_session_search({"query": "auth"})
        assert captured.get("read_only") is True


class TestShimRegistration:
    def test_both_shims_defined_by_default(self, tmp_hermes_home):
        names = [name for name, _desc, _fn in _stateless_shim_defs()]
        assert names == ["memory", "session_search"]

    def test_memory_shim_respects_config_disable(self, tmp_hermes_home, monkeypatch):
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg, "load_config", lambda *a, **k: {"memory": {"memory_enabled": False}}
        )
        names = [name for name, _desc, _fn in _stateless_shim_defs()]
        assert "memory" not in names
        assert "session_search" in names

    def test_agent_loop_refusal_stays_intact_for_other_callers(self):
        # The shims must NOT weaken the generic dispatcher: a stateless
        # handle_function_call("memory", ...) still refuses.
        from model_tools import handle_function_call

        out = handle_function_call("memory", {"action": "add", "content": "x"})
        assert "must be handled by the agent loop" in out
