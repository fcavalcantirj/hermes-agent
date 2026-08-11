"""Tests for the knock flow — owner-approved onboarding of unknown DM users.

When ``<PLATFORM>_PAIRING_APPROVERS`` is configured, an unknown DM no longer
falls into the allowlist-forced "ignore": the user is held with a polite
message, the approvers get an approve/deny prompt, and a grant lands in the
PairingStore (mirrored into the allowlist) with no gateway restart.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_store(tmp_path):
    with patch("gateway.pairing.PAIRING_DIR", tmp_path):
        return PairingStore()


# ---------------------------------------------------------------------------
# PairingStore knock records
# ---------------------------------------------------------------------------


class TestKnockStore:
    def test_create_and_read_pending_knock(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.knock_pending("telegram", "111") is None
        created = store.create_knock("telegram", "111", "Ana", "111", "oi, quero saber dos planos")
        assert created is True
        rec = store.knock_pending("telegram", "111")
        assert rec["user_name"] == "Ana"
        assert rec["chat_id"] == "111"
        assert rec["message"] == "oi, quero saber dos planos"
        assert rec["prompts"] == []

    def test_duplicate_knock_not_recreated(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.create_knock("telegram", "111", "Ana", "111", "first") is True
        assert store.create_knock("telegram", "111", "Ana", "111", "second") is False
        assert store.knock_pending("telegram", "111")["message"] == "first"

    def test_add_knock_prompt_refs(self, tmp_path):
        store = _make_store(tmp_path)
        store.create_knock("telegram", "111", "Ana", "111", "oi")
        store.add_knock_prompt("telegram", "111", "555", 42)
        store.add_knock_prompt("telegram", "111", "666", 43)
        rec = store.knock_pending("telegram", "111")
        assert {"chat_id": "555", "message_id": 42} in rec["prompts"]
        assert {"chat_id": "666", "message_id": 43} in rec["prompts"]

    def test_approve_knock_grants_and_returns_record(self, tmp_path):
        store = _make_store(tmp_path)
        store.create_knock("telegram", "111", "Ana", "111", "oi")
        record = store.approve_knock("telegram", "111", approved_by="152")
        assert record["user_name"] == "Ana"
        assert record["message"] == "oi"
        assert store.is_approved("telegram", "111") is True
        assert store.knock_pending("telegram", "111") is None

    def test_deny_knock_records_denial(self, tmp_path):
        store = _make_store(tmp_path)
        store.create_knock("telegram", "111", "Ana", "111", "oi")
        record = store.deny_knock("telegram", "111", denied_by="152")
        assert record["user_name"] == "Ana"
        assert store.is_approved("telegram", "111") is False
        assert store.is_knock_denied("telegram", "111") is True
        assert store.knock_pending("telegram", "111") is None

    def test_denied_user_cannot_knock_again(self, tmp_path):
        store = _make_store(tmp_path)
        store.create_knock("telegram", "111", "Ana", "111", "oi")
        store.deny_knock("telegram", "111")
        assert store.create_knock("telegram", "111", "Ana", "111", "de novo") is False
        assert store.knock_pending("telegram", "111") is None

    def test_approve_unknown_knock_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.approve_knock("telegram", "999") is None
        assert store.deny_knock("telegram", "999") is None

    def test_approve_knock_mirrors_allowlist_when_configured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "152")
        store = _make_store(tmp_path)
        store.create_knock("telegram", "111", "Ana", "111", "oi")
        saved = {}
        with patch("hermes_cli.config.save_env_value", side_effect=lambda k, v: saved.update({k: v})):
            store.approve_knock("telegram", "111")
        assert saved.get("TELEGRAM_ALLOWED_USERS") == "152,111"


# ---------------------------------------------------------------------------
# Unauthorized-DM behavior resolution
# ---------------------------------------------------------------------------


class _AuthzHolder:
    """Bare object carrying the mixin methods without full runner setup."""

    config = None

    _get_unauthorized_dm_behavior = GatewayRunner._get_unauthorized_dm_behavior
    _knock_approvers = GatewayRunner._knock_approvers
    _adapter_dm_policy = lambda self, platform, profile=None: ""


class TestKnockBehavior:
    def test_approvers_env_turns_ignore_into_knock(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "152")
        monkeypatch.setenv("TELEGRAM_PAIRING_APPROVERS", "152,715")
        holder = _AuthzHolder()
        assert holder._get_unauthorized_dm_behavior(Platform.TELEGRAM) == "knock"

    def test_no_approvers_keeps_ignore_with_allowlist(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "152")
        monkeypatch.delenv("TELEGRAM_PAIRING_APPROVERS", raising=False)
        holder = _AuthzHolder()
        assert holder._get_unauthorized_dm_behavior(Platform.TELEGRAM) == "ignore"

    def test_knock_approvers_parsed_and_stripped(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_PAIRING_APPROVERS", " 152 , 715 ,")
        holder = _AuthzHolder()
        assert holder._knock_approvers(Platform.TELEGRAM) == ["152", "715"]

    def test_no_platform_means_no_knock(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_PAIRING_APPROVERS", "152")
        holder = _AuthzHolder()
        assert holder._knock_approvers(None) == []


# ---------------------------------------------------------------------------
# Gateway knock request path
# ---------------------------------------------------------------------------


class _KnockAdapter:
    def __init__(self):
        self.sent = []
        self.prompts = []

    async def send(self, chat_id, text, **kwargs):
        self.sent.append((str(chat_id), text))

    async def send_knock_prompt(self, chat_id, *, user_id, user_name, message):
        self.prompts.append((str(chat_id), user_id, user_name, message))
        return (str(chat_id), 1000 + len(self.prompts))


def _make_knock_runner(store):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = _KnockAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.pairing_store = store
    runner.pairing_stores = {}
    return runner, adapter


def _stranger_event(text="oi, tudo bem?", user_id="9999"):
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id=user_id, chat_type="dm",
        user_id=user_id, user_name="Ana",
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


@pytest.mark.asyncio
async def test_unknown_dm_triggers_knock(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "152")
    monkeypatch.setenv("TELEGRAM_PAIRING_APPROVERS", "152,715")
    store = _make_store(tmp_path)
    runner, adapter = _make_knock_runner(store)
    event = _stranger_event()

    await runner._handle_knock_request(event, event.source)

    # Stranger got exactly one hold message.
    assert len(adapter.sent) == 1
    assert adapter.sent[0][0] == "9999"
    # Both approvers were prompted, and the prompt refs were recorded.
    assert [p[0] for p in adapter.prompts] == ["152", "715"]
    rec = store.knock_pending("telegram", "9999")
    assert rec["message"] == "oi, tudo bem?"
    assert len(rec["prompts"]) == 2


@pytest.mark.asyncio
async def test_repeat_messages_while_pending_stay_silent(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_PAIRING_APPROVERS", "152")
    store = _make_store(tmp_path)
    runner, adapter = _make_knock_runner(store)
    event = _stranger_event()

    await runner._handle_knock_request(event, event.source)
    await runner._handle_knock_request(event, event.source)

    assert len(adapter.sent) == 1
    assert len(adapter.prompts) == 1


@pytest.mark.asyncio
async def test_denied_user_is_silent(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_PAIRING_APPROVERS", "152")
    store = _make_store(tmp_path)
    store.create_knock("telegram", "9999", "Ana", "9999", "oi")
    store.deny_knock("telegram", "9999", denied_by="152")
    runner, adapter = _make_knock_runner(store)
    event = _stranger_event()

    await runner._handle_knock_request(event, event.source)

    assert adapter.sent == []
    assert adapter.prompts == []


@pytest.mark.asyncio
async def test_hold_message_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_PAIRING_APPROVERS", "152")
    monkeypatch.setenv("HERMES_KNOCK_HOLD_MESSAGE", "Um momento — já te atendo. 🙏")
    store = _make_store(tmp_path)
    runner, adapter = _make_knock_runner(store)
    event = _stranger_event()

    await runner._handle_knock_request(event, event.source)

    assert adapter.sent[0][1] == "Um momento — já te atendo. 🙏"


class _PlainAdapter:
    """Adapter with no send_knock_prompt — exercises the plain-text fallback."""

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, **kwargs):
        self.sent.append((str(chat_id), text))


@pytest.mark.asyncio
async def test_knock_without_prompt_capable_adapter_falls_back_to_text(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_PAIRING_APPROVERS", "152")
    store = _make_store(tmp_path)
    runner, _ = _make_knock_runner(store)
    adapter = _PlainAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    event = _stranger_event()

    await runner._handle_knock_request(event, event.source)

    # hold message + one plain-text approver notification
    assert len(adapter.sent) == 2
    assert adapter.sent[1][0] == "152"
