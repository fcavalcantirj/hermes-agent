#!/usr/bin/env python3
"""delegate_coder.py — DasBrowCoder's coding delegate (M4 v0).

One real coding task per invocation, per the kanban ownership contract: the brain
owns lifecycle/acceptance; this delegate is an input lane; its diff is untrusted
until the deterministic guard (and later the checker) says otherwise.

Mechanics:
  - sequential-only: a non-blocking flock on ~/.dasbrowcoder/delegate.lock (exit 3
    if another run is live) — never two SDK sessions at once on this Pi
  - fresh SDK session per task (Ralph-style), cwd = target repo
  - GOLDEN RULES injected verbatim into the system prompt append
  - branch-only: work happens on dasbrow/<slug>-<ts>; main must not move
  - JSONL transcript at ~/.dasbrowcoder/runs/<ts>-<slug>.jsonl with a terminal
    verdict line even on crash
  - after the run: golden_guard.py; one bounded fix round on RED
  - stdout = the verdict JSON (the brain relays it, honest labels)

Subscription lane: no ANTHROPIC_API_KEY may be present (fail-closed, same rule as
the runtime). SDK over CLI: uses claude_agent_sdk.query(), never shells `claude`.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(os.environ.get("DASBROW_HOME", "~/.dasbrowcoder")).expanduser()
# gopls (the LSP plugin's server) + the guard's gate binaries live in ~/go/bin;
# the gateway unit's PATH lacks it — extend once at import, same as the guard.
os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(Path.home() / "go" / "bin")
LOCK_PATH = HOME / "delegate.lock"
RUNS_DIR = HOME / "runs"
RULES_PATH = HOME / "GOLDEN-RULES.md"
GUARD_PATH = HOME / "golden_guard.py"
MAX_FIX_ROUNDS = 1
MAX_WALL_SECONDS = int(os.environ.get("DELEGATE_MAX_SECONDS", "1800"))  # hard kill

CONTRACT = """\
## Delegated coding task — operating contract (non-negotiable)

- You are DasBrowCoder's coding delegate working alone in this repository.
- Work ONLY on the current branch. NEVER checkout, merge, push or commit to main.
- TDD: failing test first, minimum code to green, then refactor. Commit in small
  steps on this branch with clear messages.
- Read SPEC.md first if it exists; the spec precedes the code (API-first).
- Long command output floods your context: redirect it to a file and `tail -20`
  the summary (e.g. `go test ./... > /tmp/t.out 2>&1; tail -20 /tmp/t.out`).
- LSP (gopls) is active: heed the diagnostics attached to your edits before
  declaring anything done, and use LSP references before refactoring a shared
  symbol — don't grep when the language server can answer precisely.
- The GOLDEN RULES below govern everything. If a rule blocks you, say so in your
  final message; never silently work around it.
- Finish with a short summary: what changed, what is verified (label [REAL]/[TEST]/
  [UNVERIFIED]), what is blocked.
"""


# ---------- small pure helpers (unit-tested) ----------

def slugify(task: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "task"


def branch_name(task: str, ts: str) -> str:
    return f"dasbrow/{slugify(task)}-{ts}"


def build_prompt(task: str) -> str:
    return f"{CONTRACT}\n## The task\n\n{task}\n"


def build_options_fields(repo: Path, rules_text: str) -> dict:
    """ClaudeAgentOptions fields — plain dict so tests assert without the SDK.
    Mirrors the runtime's build_option_fields() shape (claude_agent_sdk_session.py)."""
    return {
        "cwd": str(repo),
        # Autonomous coding needs to run `go test`/`go get`/git non-interactively.
        # acceptEdits denies un-allowlisted Bash headless (same wall the briefing
        # hit). bypassPermissions is the intended mode for an unattended coder —
        # scoped safe here: cwd is one repo, work is branch-only, and the guard
        # asserts main never moved. It is NOT a chat session.
        "permission_mode": "bypassPermissions",
        "system_prompt": {
            "type": "preset",
            "preset": "claude_code",
            "append": rules_text,
        },
        # LSP = the plugin-backed language-server tool (gopls): navigation ops
        # need it allowlisted; diagnostics-on-edit ride Edit results regardless.
        "allowed_tools": ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "LSP"],
        # The SDK is hermetic by default (setting_sources=None → no user
        # settings, therefore NO plugins and no LSP). "user" loads
        # ~/.claude/settings.json + installed plugins (gopls-lsp); permissions
        # stay governed by permission_mode above.
        "setting_sources": ["user"],
        "mcp_servers": {},
    }


def usage_from_events(events: list[dict]) -> dict:
    """Usage totals for the run.

    Authoritative source: the SDK's ResultMessage (dataclass-shaped event with
    cumulative usage + num_turns). Fallback for raw-CLI-shaped events (per-message
    usage): message-id-deduped sums — one API message spans multiple JSONL lines.
    """
    for ev in reversed(events):
        if ev.get("_type") == "ResultMessage" and isinstance(ev.get("usage"), dict):
            u = ev["usage"]
            return {
                "api_calls": ev.get("num_turns", 0),
                "output_tokens": u.get("output_tokens", 0),
                "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": u.get("cache_read_input_tokens", 0),
                "total_cost_usd": ev.get("total_cost_usd"),
                "duration_api_ms": ev.get("duration_api_ms"),
            }
    seen: dict[str, dict] = {}
    for ev in events:
        msg = ev.get("message") or {}
        u = msg.get("usage") or {}
        if u and msg.get("id"):
            seen[msg["id"]] = u
    return {
        "api_calls": len(seen),
        "output_tokens": sum(v.get("output_tokens", 0) for v in seen.values()),
        "cache_creation_input_tokens": sum(
            v.get("cache_creation_input_tokens", 0) for v in seen.values()
        ),
        "cache_read_input_tokens": sum(
            v.get("cache_read_input_tokens", 0) for v in seen.values()
        ),
    }


def restore_subscription_env() -> None:
    """Nested-session reality (the Claudius culture-TL;DR lesson, relearned live
    2026-07-15): Claude Code STRIPS CLAUDE_CODE_OAUTH_TOKEN from tool
    subprocesses, so a delegate spawned by the brain's Bash tool starts
    credential-less and dies in ~90ms with zero API calls. Restore the token
    deterministically from the mode-600 claude.env (single source on the box)
    and scrub the nesting markers so the child CLI starts clean."""
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        envfile = HOME / "claude.env"
        if envfile.is_file():
            for line in envfile.read_text().splitlines():
                if line.startswith("CLAUDE_CODE_OAUTH_TOKEN="):
                    val = line.split("=", 1)[1].strip().strip('"')
                    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = val
                    break
    for var in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
        os.environ.pop(var, None)


# ---------- git helpers ----------

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def rev(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", ref)


def create_branch(repo: Path, name: str) -> None:
    _git(repo, "checkout", "-q", "-b", name)


def current_branch(repo: Path) -> str:
    return _git(repo, "branch", "--show-current")


# ---------- SDK run (isolated seam; mocked in tests) ----------

def _event_to_dict(message) -> dict:
    if dataclasses.is_dataclass(message):
        d = dataclasses.asdict(message)
    else:
        d = {"repr": repr(message)}
    d["_type"] = type(message).__name__
    return d


async def _run_sdk_async(prompt: str, fields: dict, jsonl_path: Path) -> list[dict]:
    from claude_agent_sdk import ClaudeAgentOptions, query

    events: list[dict] = []
    with jsonl_path.open("a") as fh:
        async for message in query(prompt=prompt, options=ClaudeAgentOptions(**fields)):
            ev = _event_to_dict(message)
            events.append(ev)
            fh.write(json.dumps(ev, default=str) + "\n")
            fh.flush()
    return events


def run_sdk(prompt: str, fields: dict, jsonl_path: Path) -> list[dict]:
    async def _bounded():
        # Hard wall-clock kill (E2B-style lifetime cap): a hung/looping session
        # must not run unbounded on the subscription.
        return await asyncio.wait_for(
            _run_sdk_async(prompt, fields, jsonl_path), timeout=MAX_WALL_SECONDS
        )
    return asyncio.run(_bounded())


# ---------- guard ----------

def run_guard(repo: Path, jsonl: Path, budget: int | None) -> dict:
    cmd = [sys.executable, str(GUARD_PATH), "--repo", str(repo), "--jsonl", str(jsonl)]
    if budget is not None:
        cmd += ["--budget-tokens", str(budget)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"verdict": "ERROR", "reason": f"guard output unparsable: {proc.stdout[-500:]}"}


def write_evidence(repo: Path, guard: dict) -> None:
    """Evidence pack for the read-only checker (SWE-Review: reviewers with
    structured evidence beat diff-only): guard report + diff vs main, as files
    the checker can Read/Grep. Best-effort — evidence must never fail a run."""
    try:
        ev = repo / ".dasbrow"
        ev.mkdir(exist_ok=True)
        (ev / "guard-report.json").write_text(json.dumps(guard, indent=2) + "\n")
        diff = subprocess.run(["git", "diff", "main...HEAD"], cwd=repo,
                              capture_output=True, text=True, timeout=60).stdout
        (ev / "diff-vs-main.patch").write_text(diff)
    except Exception:  # noqa: BLE001
        pass


# ---------- fire-time merge grant (W1) ----------

def _default_merge_runner(message: str, repo: Path, branch: str, jsonl: Path) -> dict:
    """Fire-time merge channel. merge_branch.py is deployed beside us in HOME;
    library call (not subprocess) because we already hold the flock its CLI
    takes. The message is Felipe's VERBATIM — the merge tool parses the grant
    and enforces every gate itself; nothing is decided here."""
    sys.path.insert(0, str(HOME))
    import merge_branch  # noqa: PLC0415 — lazy: plain runs never need it
    return merge_branch.run_merge(message, repo, branch, jsonl=jsonl)


# ---------- orchestration ----------

def run_task(
    task: str,
    repo: Path,
    budget: int | None = None,
    sdk_runner=run_sdk,
    guard_runner=run_guard,
    grant_message: str | None = None,
    merge_runner=None,
) -> dict:
    ts = time.strftime("%Y%m%d-%H%M%S")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = RUNS_DIR / f"{ts}-{slugify(task)}.jsonl"
    verdict: dict = {
        "verdict": "FAIL",
        "reason": "crashed before completion",
        "task": task,
        "repo": str(repo),
        "jsonl": str(jsonl),
    }
    t0 = time.time()
    try:
        rules_text = RULES_PATH.read_text()
        main_before = rev(repo, "main")
        branch = branch_name(task, ts)
        create_branch(repo, branch)
        verdict.update({"branch": branch, "main_sha_before": main_before})

        fields = build_options_fields(repo, rules_text + "\n\n" + CONTRACT)
        events = sdk_runner(build_prompt(task), fields, jsonl)
        usage = usage_from_events(events)

        guard = guard_runner(repo, jsonl, budget)
        write_evidence(repo, guard)
        fix_rounds = 0
        while guard.get("verdict") == "RED" and fix_rounds < MAX_FIX_ROUNDS:
            fix_rounds += 1
            fix_prompt = (
                f"{CONTRACT}\n## Fix round {fix_rounds}\n\n"
                "The deterministic golden-rules guard is RED on this branch. "
                "Fix ONLY what the report below names, keeping all rules:\n\n"
                f"```json\n{json.dumps(guard, indent=2)}\n```\n"
            )
            events = sdk_runner(fix_prompt, fields, jsonl)
            u2 = usage_from_events(events)
            for k in usage:
                usage[k] += u2.get(k, 0)
            guard = guard_runner(repo, jsonl, budget)

        main_after = rev(repo, "main")
        main_moved = main_after != main_before
        on_branch = current_branch(repo) == branch
        ok = guard.get("verdict") == "GREEN" and not main_moved and on_branch
        verdict.update({
            "verdict": "PASS" if ok else "FAIL",
            "reason": ("guard GREEN, branch-only respected" if ok else
                       "main moved" if main_moved else
                       f"left branch (on '{current_branch(repo)}')" if not on_branch else
                       "guard RED after fix round"),
            "guard": guard,
            "fix_rounds": fix_rounds,
            "main_moved": main_moved,
            "usage": usage,
        })
        # Fire-time merge grant: only on PASS, and only when a grant message was
        # relayed. The merge tool re-parses the grant and re-runs the guard at
        # merge time — a merge failure never eats the task verdict.
        if grant_message and verdict["verdict"] == "PASS":
            runner = merge_runner or _default_merge_runner
            try:
                verdict["merge"] = runner(grant_message, repo, branch, jsonl)
            except Exception as exc:  # noqa: BLE001
                verdict["merge"] = {"verdict": "FAIL",
                                    "reason": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 — verdict line must always exist
        verdict.update({"verdict": "FAIL", "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        verdict["duration_s"] = round(time.time() - t0, 1)
        with jsonl.open("a") as fh:
            fh.write(json.dumps({"_type": "DelegateVerdict", **verdict}, default=str) + "\n")
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--budget-tokens", type=int)
    ap.add_argument("--grant-message",
                    help="Felipe's message VERBATIM — the merge tool parses the "
                         "grant deterministically; absent = drafts stay on the "
                         "branch, exactly as always")
    args = ap.parse_args()

    # Fail-closed subscription rule — same contract as the runtime.
    if os.environ.get("ANTHROPIC_API_KEY"):
        print(json.dumps({"verdict": "FAIL",
                          "reason": "ANTHROPIC_API_KEY present — refusing (subscription lane only)"}))
        return 2
    restore_subscription_env()
    # Positive lane assert: the credential must BE a setup-token (sk-ant-oat01-),
    # not merely "not a metered key" (cabinlab/litellm-claude-code pattern).
    oat = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if not oat:
        print(json.dumps({"verdict": "FAIL",
                          "reason": "no CLAUDE_CODE_OAUTH_TOKEN (env stripped and claude.env missing) — cannot start"}))
        return 2
    if oat and not oat.startswith("sk-ant-oat01-"):
        print(json.dumps({"verdict": "FAIL",
                          "reason": "CLAUDE_CODE_OAUTH_TOKEN is not a sk-ant-oat01- setup-token — refusing"}))
        return 2

    repo = Path(args.repo).expanduser()
    if not (repo / ".git").exists():
        print(json.dumps({"verdict": "FAIL", "reason": f"not a git repo: {repo}"}))
        return 3

    HOME.mkdir(parents=True, exist_ok=True)
    lock_fh = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"verdict": "BUSY",
                          "reason": "another delegate run is in progress (sequential-only)"}))
        return 3

    try:
        verdict = run_task(args.task, repo, args.budget_tokens,
                           grant_message=args.grant_message)
        print(json.dumps(verdict, indent=2, default=str))
        return 0 if verdict["verdict"] == "PASS" else 2
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


if __name__ == "__main__":
    sys.exit(main())
