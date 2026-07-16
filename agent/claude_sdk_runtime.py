"""claude-agent-sdk runtime — the subscription-Claude agent-loop path.

The structural twin of ``agent/codex_runtime.py``'s app-server path: hands the
entire turn to Anthropic's official ``claude-agent-sdk`` (which drives the
Claude Code CLI's own agent loop under **subscription OAuth** — never a
metered API key) and projects its typed message stream back into Hermes'
messages list so transcript persistence and recall keep working. GitHub
issue #25267.

* ``run_claude_agent_sdk_turn`` — drives one turn through a lazily-created
  ``ClaudeAgentSdkSession`` (used when ``agent.api_mode == "claude_agent_sdk"``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cap per persona/memory source so the append can't blow the context budget
# (Hermes' native files are hard-capped anyway; the soul file is ours).
_APPEND_SOURCE_MAX_CHARS = 8000


def _read_capped(path: str, cap: int = _APPEND_SOURCE_MAX_CHARS) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()[:cap].strip()
    except OSError:
        return ""


def build_system_prompt_append() -> Optional[str]:
    """Compose the system-prompt append for the SDK session.

    Sources, in order:
      1. An operator-owned persona/soul file (HERMES_CLAUDE_SDK_APPEND_FILE)
         — identity lives here; without it the agent introduces itself as
         plain Claude Code.
      2. Hermes' native memory files (~/.hermes/USER.md, MEMORY.md) — the
         learning loop's bounded sticky notes, injected whole so the SDK
         brain actually sees what Hermes has learned. (Hermes' own prompt
         composer is bypassed on this runtime; this is its replacement.)

    Read at session creation: edits apply on the next session (retire or
    gateway restart), not mid-session.
    """
    parts: list[str] = []
    soul_path = os.environ.get("HERMES_CLAUDE_SDK_APPEND_FILE", "").strip()
    if soul_path:
        soul = _read_capped(soul_path)
        if soul:
            parts.append(soul)
        else:
            logger.warning(
                "HERMES_CLAUDE_SDK_APPEND_FILE=%s is set but unreadable/empty",
                soul_path,
            )
    # ONE canonical location: the memory tool's own memories/ dir (the same
    # store the stateless MCP shim writes through — #26567). Reading the
    # HERMES_HOME root here was a path bug: tool writes landed where this
    # append never looked.
    try:
        from tools.memory_tool import get_memory_dir

        memories_dir = str(get_memory_dir())
    except Exception:  # pragma: no cover - defensive fallback, same layout
        hermes_home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
        memories_dir = os.path.join(hermes_home, "memories")
    for filename, label in (("USER.md", "About the user"), ("MEMORY.md", "Working memory")):
        content = _read_capped(os.path.join(memories_dir, filename))
        if content:
            parts.append(f"## {label} (Hermes memory — curated across sessions)\n{content}")
    return "\n\n".join(parts) or None


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def _record_claude_sdk_usage(agent, turn) -> dict[str, Any]:
    """Translate SDK ResultMessage usage into Hermes accounting.

    The SDK reports Anthropic-shaped usage: input_tokens, output_tokens,
    cache_read_input_tokens, cache_creation_input_tokens. Billing is
    subscription-included by construction (the SDK authenticates with the
    Claude subscription; there is no per-token invoice on this path)."""
    agent.session_api_calls += 1

    usage = getattr(turn, "token_usage_last", None)
    if not isinstance(usage, dict) or not usage:
        if agent._session_db and agent.session_id:
            try:
                if not agent._session_db_created:
                    agent._ensure_db_session()
                agent._session_db.update_token_counts(
                    agent.session_id,
                    model=agent.model,
                    billing_provider=agent.provider,
                    billing_base_url=agent.base_url,
                    billing_mode="subscription_included",
                    api_call_count=1,
                )
            except Exception as exc:
                logger.debug(
                    "claude-sdk api-call persistence failed (session=%s): %s",
                    agent.session_id, exc,
                )
        return {}

    from agent.usage_pricing import CanonicalUsage

    input_tokens = _coerce_usage_int(usage.get("input_tokens"))
    output_tokens = _coerce_usage_int(usage.get("output_tokens"))
    cache_read_tokens = _coerce_usage_int(usage.get("cache_read_input_tokens"))
    cache_write_tokens = _coerce_usage_int(
        usage.get("cache_creation_input_tokens")
    )

    canonical_usage = CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=0,
        raw_usage=usage,
    )
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = canonical_usage.total_tokens
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": 0,
    }

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        try:
            compressor.update_from_response(usage_dict)
        except Exception:
            logger.debug("claude-sdk usage update failed", exc_info=True)

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens

    agent.session_cost_status = "included"
    agent.session_cost_source = "claude-subscription"

    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.update_token_counts(
                agent.session_id,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=0,
                cost_status="included",
                cost_source="claude-subscription",
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included",
                model=agent.model,
                api_call_count=1,
            )
        except Exception as exc:
            logger.debug(
                "claude-sdk token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, exc,
            )

    return {
        **usage_dict,
        "last_prompt_tokens": prompt_tokens,
        "estimated_cost_usd": None,
        "cost_status": "included",
        "cost_source": "claude-subscription",
    }


def run_claude_agent_sdk_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """claude-agent-sdk runtime path. Hands the entire turn to the SDK's
    agent loop and projects its messages back into Hermes' list.

    Called from run_conversation() when agent.api_mode == "claude_agent_sdk".
    Returns the same dict shape as the chat_completions path."""
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession

    if not hasattr(agent, "_claude_sdk_session") or agent._claude_sdk_session is None:
        from agent.runtime_cwd import resolve_agent_cwd

        cwd = getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())
        try:
            from tools.terminal_tool import _get_approval_callback
            approval_callback = _get_approval_callback()
        except Exception:
            approval_callback = None

        def _on_tool_started(tool_name: str, preview: str, args: dict) -> None:
            progress_callback = getattr(agent, "tool_progress_callback", None)
            if progress_callback is None:
                return
            try:
                progress_callback("tool.started", tool_name, preview, args)
            except Exception:
                logger.debug(
                    "claude-sdk tool-progress callback raised", exc_info=True
                )

        agent._claude_sdk_session = ClaudeAgentSdkSession(
            cwd=cwd,
            model=getattr(agent, "model", None) or None,
            approval_callback=approval_callback,
            on_tool_started=_on_tool_started,
            system_prompt_append=build_system_prompt_append(),
            hermes_session_id=getattr(agent, "session_id", None),
        )

    # NOTE: the user message is ALREADY appended to messages by the standard
    # run_conversation() flow before the early return reaches us. Do NOT
    # append again — that would duplicate. (Same contract as codex_runtime.)

    try:
        turn = agent._claude_sdk_session.run_turn(user_input=user_message)
    except Exception as exc:
        logger.exception("claude-agent-sdk turn failed")
        try:
            agent._claude_sdk_session.close()
        except Exception:
            pass
        agent._claude_sdk_session = None
        return {
            "final_response": f"claude-agent-sdk turn failed: {exc}",
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
        }

    if getattr(turn, "should_retire", False):
        logger.warning(
            "claude-agent-sdk session retired (turn error: %s)", turn.error
        )
        try:
            agent._claude_sdk_session.close()
        except Exception:
            pass
        agent._claude_sdk_session = None

    if turn.projected_messages:
        messages.extend(turn.projected_messages)
        # Early-return path bypasses conversation_loop's per-step persistence;
        # flush the new projected rows ourselves (idempotent via the intrinsic
        # _DB_PERSISTED_MARKER — the user turn was flushed at turn start).
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug(
                    "claude-sdk projected-message flush failed", exc_info=True
                )

    # Counter ticks — _turns_since_memory/_user_turn_count are incremented by
    # run_conversation()'s pre-loop block; only _iters_since_skill is ours.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    usage_result = _record_claude_sdk_usage(agent, turn)

    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    if (
        turn.final_text
        and not turn.interrupted
        and (should_review_memory or should_review_skills)
    ):
        # Deliberately NOT spawning the background review on this runtime:
        # the fork inherits api_mode="claude_agent_sdk" and early-returns
        # into a fresh SDK session whose tool surface has no `memory` /
        # `skill_manage` — it would burn a subscription turn and be unable
        # to write anything. The nudge counters above keep ticking so a
        # bounded replacement pass can reuse them. (#25267)
        logger.debug(
            "claude-sdk runtime: background review skipped "
            "(memory=%s, skills=%s) — the review fork cannot write on "
            "this runtime",
            should_review_memory,
            should_review_skills,
        )

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": 1,
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "error": turn.error,
        # Same persistence contract as the codex app-server path: we flushed
        # the projected rows ourselves, so the gateway must not re-write the
        # user turn (append_message has no dedup).
        "agent_persisted": True,
        "claude_sdk_session_id": turn.thread_id,
        **usage_result,
    }


__all__ = ["run_claude_agent_sdk_turn"]
