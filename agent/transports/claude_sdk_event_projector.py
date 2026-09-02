"""Projects Claude Agent SDK messages into Hermes' messages list.

The translator that lets Hermes' memory/skill review keep working under the
claude-agent-sdk runtime: it converts the SDK's typed message stream into the
standard OpenAI-shaped ``{role, content, tool_calls, tool_call_id}`` entries
that ``agent/curator.py`` already knows how to read. The structural twin of
``codex_event_projector.py`` — same contract, different wire shape.

The SDK stream (``ClaudeSDKClient.receive_response()``) yields, per turn:
  - AssistantMessage   → content blocks: TextBlock / ThinkingBlock / ToolUseBlock
  - UserMessage        → tool results echoed back: ToolResultBlock entries
  - SystemMessage      → lifecycle events (init, …) — not conversation content
  - StreamEvent        → display-only partial deltas — never materialized
  - ResultMessage      → terminal: final text, usage, cost, error subtype

Projection rules (mirrors the codex projector's invariants):
  - One AssistantMessage → ONE assistant entry (text + tool_calls together).
  - Each ToolResultBlock → one ``{role: "tool", tool_call_id, content}`` entry;
    ticks ``is_tool_iteration`` (skill-nudge counter parity).
  - ThinkingBlock text is stashed on the next assistant entry's ``reasoning``.
  - ResultMessage sets ``final_text`` (authoritative over the last text block).

This module deliberately duck-types on ``type(message).__name__`` instead of
importing ``claude_agent_sdk`` — it stays importable (and unit-testable) when
the optional SDK extra is not installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

_TOOL_RESULT_MAX_CHARS = 4000
_TOOL_RESULT_TRUNCATION_MARKER = "...[truncated]"


def _sdk_type_name(obj: Any) -> str:
    return type(obj).__name__


def _shrink_json_strings(value: Any, max_chars: int) -> Any:
    """Return a shape-preserving copy with oversized string leaves shortened."""

    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        keep = max(0, max_chars - len(_TOOL_RESULT_TRUNCATION_MARKER))
        return value[:keep] + _TOOL_RESULT_TRUNCATION_MARKER
    if isinstance(value, list):
        return [_shrink_json_strings(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {
            key: _shrink_json_strings(item, max_chars)
            for key, item in value.items()
        }
    return value


def _truncate_tool_result_text(text: str) -> str:
    """Cap projected tool text while preserving parseable structured receipts."""

    if len(text) <= _TOOL_RESULT_MAX_CHARS:
        return text
    try:
        decoded = json.loads(text)
    except (TypeError, ValueError):
        return text[:_TOOL_RESULT_MAX_CHARS]

    # Large delegate results commonly contain one long ``summary`` followed by
    # route/billing metadata.  Head-only truncation makes the JSON invalid and
    # drops that fail-closed receipt from persisted history.  Shrink string
    # leaves progressively so the original object/list shape and metadata stay
    # parseable.  Unusually huge structural payloads retain the legacy cap.
    for max_chars in (2048, 1024, 512, 256, 128, 64):
        compact = json.dumps(
            _shrink_json_strings(decoded, max_chars),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(compact) <= _TOOL_RESULT_MAX_CHARS:
            return compact
    return text[:_TOOL_RESULT_MAX_CHARS]


def _flatten_tool_result_content(content: Any) -> str:
    """ToolResultBlock.content is str | list[dict] | None — flatten to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return _truncate_tool_result_text(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                else:
                    try:
                        parts.append(json.dumps(item, ensure_ascii=False))
                    except (TypeError, ValueError):
                        parts.append(repr(item))
            elif item is not None:
                parts.append(str(item))
        return _truncate_tool_result_text("\n".join(parts))
    return _truncate_tool_result_text(str(content))


def _format_tool_args(d: Any) -> str:
    """Format tool input as JSON the way Hermes' existing tool_calls path does."""
    if not isinstance(d, dict):
        d = {"input": d}
    try:
        return json.dumps(d, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return json.dumps({"repr": repr(d)}, ensure_ascii=False)


@dataclass
class ProjectionResult:
    """Output of projecting one SDK message.

    ``messages`` is a list because one AssistantMessage can produce one
    assistant entry and a UserMessage can carry several tool results. Empty
    list = message ignored (SystemMessage lifecycle, StreamEvent deltas)."""

    messages: list[dict] = field(default_factory=list)
    is_tool_iteration: bool = False
    final_text: Optional[str] = None  # Set when text lands / on ResultMessage
    is_result: bool = False  # True only for the terminal ResultMessage
    model: Optional[str] = None  # Model id the SDK reported for this message


class ClaudeSdkEventProjector:
    """Stateful projector consuming SDK messages in arrival order.

    Owns pending thinking text (stashed onto the next assistant entry,
    mirroring how the codex projector holds reasoning items)."""

    def __init__(self) -> None:
        self._pending_thinking: list[str] = []

    def project(self, message: Any) -> ProjectionResult:
        name = _sdk_type_name(message)
        if name == "AssistantMessage":
            return self._project_assistant(message)
        if name == "UserMessage":
            return self._project_user(message)
        if name == "ResultMessage":
            return self._project_result(message)
        # SystemMessage / StreamEvent / unknown lifecycle types: display or
        # bookkeeping only — nothing enters the conversation transcript.
        return ProjectionResult()

    # ---------- per-type projections ----------

    def _project_assistant(self, message: Any) -> ProjectionResult:
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in getattr(message, "content", None) or []:
            bname = _sdk_type_name(block)
            if bname == "TextBlock":
                text = getattr(block, "text", "") or ""
                if text:
                    text_parts.append(text)
            elif bname == "ThinkingBlock":
                thinking = getattr(block, "thinking", "") or ""
                if thinking:
                    self._pending_thinking.append(thinking)
            elif bname == "ServerToolUseBlock":
                # Server tools (web_search, web_fetch, ...) execute API-side;
                # their results arrive as ServerToolResultBlocks inside the
                # SAME assistant message, never as a {role:'tool'} echo — an
                # OpenAI-shaped tool_calls entry here would persist a
                # dangling tool_call_id that breaks replay through native
                # provider paths. The tool's textual outcome already arrives
                # in the assistant text; nothing to project here.
                pass
            elif bname == "ToolUseBlock":
                call_id = getattr(block, "id", "") or ""
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": getattr(block, "name", "") or "unknown",
                            "arguments": _format_tool_args(
                                getattr(block, "input", None) or {}
                            ),
                        },
                    }
                )
            # ToolResultBlock never appears on AssistantMessage; other block
            # types (server tool results, …) are ignored here on purpose.

        if not text_parts and not tool_calls:
            # Thinking-only messages still carry the model id.
            return ProjectionResult(model=getattr(message, "model", None) or None)

        msg: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if self._pending_thinking:
            msg["reasoning"] = "\n".join(self._pending_thinking)
            self._pending_thinking = []
        final_text = "\n".join(text_parts) if text_parts else None
        return ProjectionResult(
            messages=[msg],
            final_text=final_text,
            model=getattr(message, "model", None) or None,
        )

    def _project_user(self, message: Any) -> ProjectionResult:
        """SDK UserMessages inside a response stream carry tool results."""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            # A plain-text user echo — not conversation-new under this
            # runtime (Hermes already appended the real user turn).
            return ProjectionResult()
        out: list[dict] = []
        tool_iteration = False
        for block in content or []:
            if _sdk_type_name(block) != "ToolResultBlock":
                continue
            text = _flatten_tool_result_content(getattr(block, "content", None))
            if getattr(block, "is_error", False):
                text = f"[error] {text}" if text else "[error]"
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(block, "tool_use_id", "") or "",
                    "content": text,
                }
            )
            tool_iteration = True
        return ProjectionResult(messages=out, is_tool_iteration=tool_iteration)

    def _project_result(self, message: Any) -> ProjectionResult:
        final = getattr(message, "result", None)
        return ProjectionResult(
            final_text=final if isinstance(final, str) and final else None,
            is_result=True,
            model=getattr(message, "model", None) or None,
        )
