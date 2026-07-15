"""Tests for the claude-agent-sdk runtime (#25267).

Covers the three new modules end-to-end without requiring the optional
``claude-agent-sdk`` extra: the projector and session duck-type on class
NAMES, so local stand-in classes named like the SDK's types are the fixture.

Plant-the-failure discipline: every guard here is exercised RED first —
the auth classifier has a negative control (an ordinary error must NOT
produce the re-auth hint), and the session's error path is asserted to
retire the client rather than silently continue.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from agent.claude_sdk_runtime import run_claude_agent_sdk_turn
from agent.transports.claude_agent_sdk_session import (
    ClaudeAgentSdkSession,
    classify_auth_failure,
)
from agent.transports.claude_sdk_event_projector import (
    ClaudeSdkEventProjector,
)


# ---------- SDK stand-in types (duck-typed by class NAME) ----------


@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str
    signature: str = ""


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: Any = None
    is_error: Optional[bool] = None


@dataclass
class AssistantMessage:
    content: list
    model: str = "claude-opus-4-8"


@dataclass
class UserMessage:
    content: Any = None


@dataclass
class SystemMessage:
    subtype: str = "init"
    data: dict = field(default_factory=dict)


@dataclass
class ResultMessage:
    subtype: str = "success"
    duration_ms: int = 1
    duration_api_ms: int = 1
    is_error: bool = False
    num_turns: int = 1
    session_id: str = "sdk-session-1"
    result: Optional[str] = None
    usage: Optional[dict] = None
    uuid: Optional[str] = "uuid-1"
    errors: Optional[list] = None


# ---------- projector ----------


class TestProjector:
    def test_assistant_text(self):
        p = ClaudeSdkEventProjector()
        out = p.project(AssistantMessage(content=[TextBlock("hello")]))
        assert out.messages == [{"role": "assistant", "content": "hello"}]
        assert out.final_text == "hello"
        assert not out.is_tool_iteration

    def test_assistant_tool_use_and_thinking(self):
        p = ClaudeSdkEventProjector()
        # Thinking arrives first, stashes onto the next assistant entry.
        p.project(AssistantMessage(content=[ThinkingBlock("pondering")]))
        out = p.project(
            AssistantMessage(
                content=[ToolUseBlock(id="t1", name="Bash", input={"command": "ls"})]
            )
        )
        (msg,) = out.messages
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert msg["reasoning"] == "pondering"
        (call,) = msg["tool_calls"]
        assert call["id"] == "t1"
        assert call["function"]["name"] == "Bash"
        assert '"command": "ls"' in call["function"]["arguments"]

    def test_tool_result_projection(self):
        p = ClaudeSdkEventProjector()
        out = p.project(
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="ok")])
        )
        assert out.is_tool_iteration
        assert out.messages == [
            {"role": "tool", "tool_call_id": "t1", "content": "ok"}
        ]

    def test_tool_result_error_and_list_content(self):
        p = ClaudeSdkEventProjector()
        out = p.project(
            UserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id="t2",
                        content=[{"type": "text", "text": "boom"}],
                        is_error=True,
                    )
                ]
            )
        )
        assert out.messages[0]["content"] == "[error] boom"

    def test_tool_result_truncation(self):
        p = ClaudeSdkEventProjector()
        out = p.project(
            UserMessage(
                content=[ToolResultBlock(tool_use_id="t3", content="x" * 9000)]
            )
        )
        assert len(out.messages[0]["content"]) == 4000

    def test_result_message_sets_final_text(self):
        p = ClaudeSdkEventProjector()
        out = p.project(ResultMessage(result="the answer"))
        assert out.is_result
        assert out.final_text == "the answer"
        assert out.messages == []

    def test_lifecycle_messages_ignored(self):
        p = ClaudeSdkEventProjector()
        assert p.project(SystemMessage()).messages == []
        # A plain-text user echo must not duplicate the real user turn.
        assert p.project(UserMessage(content="hi")).messages == []


# ---------- auth classifier (with negative control) ----------


class TestAuthClassifier:
    def test_auth_failure_produces_hint(self):
        hint = classify_auth_failure("HTTP 401 unauthorized: oauth token expired")
        assert hint is not None
        assert "setup-token" in hint

    def test_negative_control_ordinary_error_no_hint(self):
        # RED-first: an unrelated failure must surface verbatim, never as a
        # re-auth redirect.
        assert classify_auth_failure("connection reset by peer") is None
        assert classify_auth_failure("") is None


# ---------- session (fake client) ----------


class _FakeClient:
    """Stub ClaudeSDKClient: async surface, scripted message stream."""

    def __init__(self, options=None, script=None, connect_exc=None):
        self.options = options
        self._script = script or []
        self._connect_exc = connect_exc
        self.queried: list[str] = []
        self.disconnected = False
        self.interrupted = False

    async def connect(self):
        if self._connect_exc is not None:
            raise self._connect_exc

    async def query(self, text):
        self.queried.append(text)

    async def receive_response(self):
        for message in self._script:
            yield message

    async def interrupt(self):
        self.interrupted = True

    async def disconnect(self):
        self.disconnected = True


def _make_session(script=None, connect_exc=None, **kwargs):
    holder = {}

    def factory(options=None):
        holder["client"] = _FakeClient(
            options=options, script=script, connect_exc=connect_exc
        )
        return holder["client"]

    session = ClaudeAgentSdkSession(
        cwd="/tmp", model="claude-opus-4-8", client_factory=factory, **kwargs
    )
    return session, holder


class TestSession:
    def test_happy_turn(self):
        script = [
            AssistantMessage(
                content=[ToolUseBlock(id="t1", name="Read", input={"file_path": "/x"})]
            ),
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="data")]),
            AssistantMessage(content=[TextBlock("done reading")]),
            ResultMessage(
                result="done reading",
                usage={"input_tokens": 10, "output_tokens": 5},
            ),
        ]
        session, holder = _make_session(script=script)
        try:
            turn = session.run_turn("read /x please")
        finally:
            session.close()
        assert turn.error is None
        assert turn.final_text == "done reading"
        assert turn.tool_iterations == 1
        assert turn.token_usage_last == {"input_tokens": 10, "output_tokens": 5}
        assert turn.thread_id == "sdk-session-1"
        # assistant(tool_call) + tool + assistant(text)
        assert [m["role"] for m in turn.projected_messages] == [
            "assistant", "tool", "assistant",
        ]
        assert holder["client"].queried == ["read /x please"]
        assert not turn.should_retire

    def test_sdk_error_result_surfaces(self):
        script = [ResultMessage(subtype="error_max_turns", is_error=False)]
        session, _ = _make_session(script=script)
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert "error_max_turns" in turn.error

    def test_auth_error_marks_retire(self):
        script = [
            ResultMessage(
                subtype="success",
                is_error=True,
                errors=["401 unauthorized: invalid bearer token"],
            )
        ]
        session, _ = _make_session(script=script)
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.should_retire
        assert "setup-token" in (turn.error or "")

    def test_connect_failure_fails_closed(self):
        session, _ = _make_session(connect_exc=RuntimeError("not logged in"))
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.should_retire
        assert turn.error is not None

    def test_option_fields_shape(self):
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        options = holder["client"].options
        assert options["model"] == "claude-opus-4-8"
        assert options["system_prompt"]["preset"] == "claude_code"
        assert "hermes-tools" in options["mcp_servers"]
        mcp = options["mcp_servers"]["hermes-tools"]
        assert mcp["args"] == ["-m", "agent.transports.hermes_tools_mcp_server"]
        # Hard rule: a metered key never reaches any child of this runtime.
        assert "ANTHROPIC_API_KEY" not in (mcp.get("env") or {})
        assert options["permission_mode"] in {
            "acceptEdits", "default", "bypassPermissions",
        }

    def test_metered_key_scrubbed_from_mcp_env(self, monkeypatch):
        # RED-first: with the ambient var set, the builder must scrub it.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        fields = session.build_option_fields()
        assert "ANTHROPIC_API_KEY" not in fields["mcp_servers"]["hermes-tools"]["env"]

    def test_metered_key_refuses_startup_fail_closed(self, monkeypatch):
        # The hard rule enforced at the front door: a present metered key
        # must abort the REAL runtime startup path, never silently rebill.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
        monkeypatch.delenv("HERMES_CLAUDE_SDK_ALLOW_API_KEY", raising=False)
        session = ClaudeAgentSdkSession(cwd="/tmp")  # no factory → real path
        turn = session.run_turn("hi")
        assert turn.should_retire
        assert "ANTHROPIC_API_KEY" in (turn.error or "")


# ---------- runtime glue ----------


def _make_turn(**overrides):
    base = dict(
        interrupted=False,
        error=None,
        thread_id="sdk-session-1",
        turn_id="uuid-1",
        projected_messages=[{"role": "assistant", "content": "SDK_ASSISTANT"}],
        tool_iterations=2,
        final_text="SDK_ASSISTANT",
        should_retire=False,
        token_usage_last={"input_tokens": 7, "output_tokens": 3},
        token_usage_total=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_agent():
    agent = MagicMock()
    agent._claude_sdk_session = MagicMock()
    agent._claude_sdk_session.run_turn.return_value = _make_turn()
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = None
    agent._session_db_created = True
    agent.session_id = "sess-1"
    agent.session_api_calls = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.context_compressor = None
    agent.model = "claude-opus-4-8"
    agent.provider = "claude-agent-sdk"
    agent.base_url = ""
    return agent


class TestRuntimeGlue:
    def test_turn_contract(self):
        agent = _make_agent()
        messages = [{"role": "user", "content": "hi"}]
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=messages,
            effective_task_id="task-1",
        )
        assert result["final_response"] == "SDK_ASSISTANT"
        assert result["completed"] is True
        assert result["agent_persisted"] is True
        assert result["cost_status"] == "included"
        assert result["cost_source"] == "claude-subscription"
        # Projected messages spliced after the (pre-appended) user turn.
        assert messages[-1]["content"] == "SDK_ASSISTANT"
        # Skill-nudge counter parity with the codex path.
        assert agent._iters_since_skill == 2

    def test_retire_closes_session(self):
        agent = _make_agent()
        agent._claude_sdk_session.run_turn.return_value = _make_turn(
            should_retire=True, error="turn timed out after 600s",
            projected_messages=[], final_text="", token_usage_last=None,
        )
        stale = agent._claude_sdk_session
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        stale.close.assert_called_once()
        assert agent._claude_sdk_session is None
        assert result["partial"] is True


# ---------- provider wiring ----------


class TestProviderWiring:
    def test_profile_registered_with_aliases(self):
        from providers import get_provider_profile

        profile = get_provider_profile("claude-agent-sdk")
        assert profile is not None
        assert profile.api_mode == "claude_agent_sdk"
        assert profile.auth_type == "oauth_external"
        assert get_provider_profile("claude-sdk") is profile
        # The anthropic profile keeps its own alias namespace untouched.
        anthropic = get_provider_profile("claude")
        assert anthropic is not None and anthropic.name == "anthropic"

    def test_runtime_resolution_short_circuit(self):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested="claude-agent-sdk")
        assert runtime["provider"] == "claude-agent-sdk"
        assert runtime["api_mode"] == "claude_agent_sdk"
        # No credential-pool machinery, no metered key.
        assert runtime["api_key"] == "claude-subscription-oauth"

    def test_api_mode_accepted_by_agent_init(self):
        from hermes_cli.runtime_provider import _parse_api_mode

        assert _parse_api_mode("claude_agent_sdk") == "claude_agent_sdk"


class TestSdkAvailabilityGate:
    def test_check_reports_missing_sdk(self, monkeypatch):
        # RED-first negative control: with the import broken, the gate must
        # fail with the install hint — never silently pass.
        import builtins

        real_import = builtins.__import__

        def _broken(name, *args, **kwargs):
            if name == "claude_agent_sdk":
                raise ImportError("No module named 'claude_agent_sdk'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _broken)
        from agent.transports.claude_agent_sdk_session import (
            check_claude_sdk_available,
        )

        ok, msg = check_claude_sdk_available()
        assert ok is False
        assert "hermes-agent[claude-agent-sdk]" in msg
