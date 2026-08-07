"""Direct-outbound delivery of claude-agent-sdk background results.

The old lane wrapped the agent's own finished answer in a synthetic empty-id
async_delegation and re-injected it into the SAME session as a fake
delegation completion; the model recognized its own text, refused to "relay"
it, and the 2026-08-06 research report never left the box. These tests pin
the replacement: ``sdk_background_result`` events are sent DIRECTLY on the
platform outbound lane — adapter.handle_message is never involved, payloads
arrive as separate messages in order, and an event with no route is requeued,
never silently dropped.
"""

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform
from gateway.run import GatewayRunner, _drain_gateway_watch_events


def _adapter(**over):
    adapter = SimpleNamespace(
        supports_async_delivery=True,
        extract_media=lambda text: ([], text),
        extract_images=lambda text: ([], text),
        send=AsyncMock(),
        send_image=AsyncMock(),
        handle_message=AsyncMock(),
    )
    for key, value in over.items():
        setattr(adapter, key, value)
    return adapter


def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={},
    )
    runner._session_source_cache = {}
    runner._thread_metadata_for_source = lambda source, mid=None: None
    return runner


def _event(**over):
    evt = {
        "type": "sdk_background_result",
        "payloads": ["Research landed — writing up.", "the full report"],
        "session_key": "agent:main:telegram:dm:12345:678",
        "parent_session_id": "sess-bg-1",
        "model": "m",
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
    }
    evt.update(over)
    return evt


def test_bg_result_main_agent_sent_outbound_never_reinjected():
    adapter = _adapter()
    runner = _runner(adapter)
    evt = _event()

    delivered = asyncio.run(runner._deliver_sdk_background_result(evt))

    assert delivered is True
    assert adapter.handle_message.call_count == 0
    assert adapter.send.call_count == 2
    for call in adapter.send.call_args_list:
        content = call.kwargs["content"]
        assert "[ASYNC DELEGATION COMPLETE" not in content
        assert "[USER IS WAITING" not in content
        assert call.kwargs["chat_id"] == "12345"


def test_bg_result_payloads_sent_in_order_without_directive():
    adapter = _adapter()
    runner = _runner(adapter)
    evt = _event(payloads=["one", "two", "three"])

    delivered = asyncio.run(runner._deliver_sdk_background_result(evt))

    assert delivered is True
    sent = [c.kwargs["content"] for c in adapter.send.call_args_list]
    assert sent == ["one", "two", "three"]
    assert not any("[USER IS WAITING" in s for s in sent)


def test_bg_result_unroutable_event_requeued_not_dropped():
    # Unparseable/empty session_key: no source -> retryable False, not None.
    adapter = _adapter()
    runner = _runner(adapter)
    assert asyncio.run(
        runner._deliver_sdk_background_result(_event(session_key=""))
    ) is False
    assert adapter.send.call_count == 0

    # Adapter for the platform missing entirely -> False too.
    runner_no_adapter = _runner(None)
    assert asyncio.run(
        runner_no_adapter._deliver_sdk_background_result(_event())
    ) is False

    # Non-push adapter (api_server-style) delivers by re-injection — no
    # deliverable route on this lane.
    non_push = _adapter(supports_async_delivery=False)
    runner_non_push = _runner(non_push)
    assert asyncio.run(
        runner_non_push._deliver_sdk_background_result(_event())
    ) is False
    assert non_push.send.call_count == 0

    # Empty payloads: nothing to send — dropped as None, never retried.
    assert asyncio.run(
        runner._deliver_sdk_background_result(_event(payloads=[]))
    ) is None


def test_bg_result_partial_failure_requeues_remaining():
    adapter = _adapter()
    adapter.send = AsyncMock(side_effect=[None, RuntimeError("flaky network")])
    runner = _runner(adapter)
    evt = _event(payloads=["one", "two", "three"])

    delivered = asyncio.run(runner._deliver_sdk_background_result(evt))

    assert delivered is False
    # The delivered prefix is trimmed so the retry sends only the remainder.
    assert evt["payloads"] == ["two", "three"]


def test_post_turn_drain_leaves_sdk_background_result_on_queue():
    # The post-turn watch drain owns only watch events; it must requeue an
    # sdk_background_result rather than consuming it (a consumed event is a
    # silently lost finished result).
    q = queue.Queue()
    evt = _event()
    q.put(evt)
    q.put({"type": "watch_match", "pattern": "x", "output": "y"})

    watch_events = _drain_gateway_watch_events(q)

    assert [e.get("type") for e in watch_events] == ["watch_match"]
    remaining = []
    while not q.empty():
        remaining.append(q.get_nowait())
    assert remaining == [evt]
