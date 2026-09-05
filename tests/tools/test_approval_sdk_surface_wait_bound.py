"""The SDK gateway approval surface stays bounded by ``approvals.timeout`` under an inherited
CLI/TUI/Desktop wait-for-answer binding, while a native attended prompt keeps waiting.

The SDK reader task, and so its ``can_use_tool`` -> ``asyncio.to_thread`` approval worker, runs in
the context of the turn that created the session (``run_coroutine_threadsafe`` captures the caller's
context), so a TUI-born SDK session carries ``prompts_wait_for_answer`` into every approval wait.
That worker is never interrupted, so the surface — not the inherited binding — picks its window.

Synchronization is by events only (prompt published, accounting window open); every worker is
released in ``finally`` so a failing assertion cannot leave one blocked.
"""

import contextlib
import contextvars
import json
import threading

import pytest

from tools import approval as mod
from tools import approval_context as ctx
from tools import approval_gateway_wait as wait_mod
from tools import approval_human_wait as human_wait
from tools import approval_sdk_gateway as sdk_gateway

APPROVAL_TIMEOUT_S = 1
WAIT_S = 30.0  # generous bound for every event wait and join
SDK_TIMEOUT = {"choice": "deny", "reason": "approval timed out — no operator response"}
CANONICAL = json.dumps({"tool_name": "Bash", "tool_input": {"command": "true"}},
                       sort_keys=True, separators=(",", ":"))
NATIVE_APPROVAL = {"command": "rm -rf build", "description": "d", "pattern_key": "dangerous",
                   "pattern_keys": ["dangerous"]}


class _Observed:
    """Per-key evidence from the real wait path: prompt published, accounting window open (with the
    ceiling it entered with), and the settle reasons that withdraw the prompt."""

    def __init__(self):
        self.published: dict[str, threading.Event] = {}
        self.window_open: dict[str, threading.Event] = {}
        self.ceilings: dict[str, float] = {}
        self.settles: dict[str, list[str]] = {}

    def event(self, table, key):
        return table.setdefault(key, threading.Event())

    def notifier(self, key):
        def notify(data):
            self.settles.setdefault(key, [])
            assert mod.register_gateway_settle(key, data["request_id"], self.settles[key].append)
            self.event(self.published, key).set()
        return notify


@pytest.fixture
def observed(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(ctx, "_get_approval_config", lambda: {"timeout": APPROVAL_TIMEOUT_S, "mode": "manual"})
    monkeypatch.setattr(ctx, "_fire_approval_hook", lambda name, **kw: None)
    obs = _Observed()
    real_window = wait_mod.human_wait_window

    @contextlib.contextmanager
    def window(session_key=None, *, wait_seconds=None):
        with real_window(session_key, wait_seconds=wait_seconds):
            with human_wait._human_wait_lock:
                obs.ceilings[session_key] = human_wait._human_wait_states[session_key].window_ceiling
            obs.event(obs.window_open, session_key).set()
            yield

    monkeypatch.setattr(wait_mod, "human_wait_window", window)
    return obs


def _inherited_turn(key, body):
    """A worker whose context carries what a TUI-born SDK session inherits: key + wait-for-answer."""
    def run():
        ctx.set_current_session_key(key)
        ctx.set_prompts_wait_for_answer()
        body()
    thread = threading.Thread(target=lambda: contextvars.Context().run(run), daemon=True)
    thread.start()
    return thread


def _sdk_wait(key, box):
    def body():
        callback = sdk_gateway.build_sdk_gateway_approval_callback()
        box["outcome"] = callback("true", "true", canonical_tool_input=CANONICAL)
    return _inherited_turn(key, body)


def _release(key, worker):
    """Withdraw every waiter for *key* (unregister wakes them), join, and drop leftover state."""
    mod.unregister_gateway_notify(key)
    if worker is not None:
        worker.join(WAIT_S)
    with mod._lock:
        mod._gateway_queues.pop(key, None)
    with human_wait._human_wait_lock:
        human_wait._human_wait_states.pop(key, None)


def _attended_ceiling():
    def read():
        ctx.set_prompts_wait_for_answer()
        return human_wait.human_wait_ceiling()
    return contextvars.Context().run(read)


def test_sdk_wait_under_inherited_binding_expires_honestly_and_releases_its_worker(observed):
    key, box, worker = "sdk-bounded", {}, None
    mod.register_gateway_notify(key, observed.notifier(key))
    try:
        worker = _sdk_wait(key, box)
        assert observed.event(observed.published, key).wait(WAIT_S)
        assert observed.event(observed.window_open, key).wait(WAIT_S)
        # Accounting is fixed at the surface's bound, not the inherited unbounded window.
        assert observed.ceilings[key] == human_wait.human_wait_ceiling(wait_seconds=APPROVAL_TIMEOUT_S)
        assert observed.ceilings[key] < _attended_ceiling()

        worker.join(WAIT_S)
        assert not worker.is_alive()
        assert box["outcome"] == SDK_TIMEOUT
        assert observed.settles[key] == ["timeout"]
        assert key not in mod._gateway_queues
    finally:
        _release(key, worker)


def test_native_attended_prompt_outlives_the_sdk_expiry(observed):
    native_key, sdk_key = "native-attended", "sdk-bounded-control"
    native_box, sdk_box, native, sdk = {}, {}, None, None
    mod.register_gateway_notify(sdk_key, observed.notifier(sdk_key))

    def native_body():
        native_box["decision"] = wait_mod._await_gateway_decision(
            native_key, observed.notifier(native_key), dict(NATIVE_APPROVAL))

    try:
        native = _inherited_turn(native_key, native_body)
        assert observed.event(observed.window_open, native_key).wait(WAIT_S)

        # The SDK wait starts after the native window opened; its own expiry proves approvals.timeout
        # has elapsed while the native prompt was open.
        sdk = _sdk_wait(sdk_key, sdk_box)
        sdk.join(WAIT_S)
        assert not sdk.is_alive()
        assert sdk_box["outcome"] == SDK_TIMEOUT

        assert native.is_alive()
        assert len(mod._gateway_queues.get(native_key, [])) == 1
        assert observed.ceilings[native_key] == _attended_ceiling()

        assert mod.resolve_gateway_approval(native_key, "once") == 1
        native.join(WAIT_S)
        assert native_box["decision"] == {"resolved": True, "choice": "once", "reason": None}
        assert observed.settles[native_key] == ["resolved"]
    finally:
        _release(native_key, native)
        _release(sdk_key, sdk)
