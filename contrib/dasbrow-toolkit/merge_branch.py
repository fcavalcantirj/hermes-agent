#!/usr/bin/env python3
"""merge_branch.py — deterministic per-task merge authorization gate (W1 v1).

Felipe's ruling (2026-07-15): human-merge stays the default, forever. A merge
happens ONLY when his own message contains an exact grant phrase — parsed HERE,
deterministically, never inferred by a model. The brain's only job is to carry
his message to this tool verbatim (fire-time via the delegate's
--grant-message, post-hoc via this CLI); this tool decides.

Mechanics:
  - grant gate: exact phrase set, own line or message edge, trailing "?" refuses
    (a question is not a grant), negation tokens on the matched line refuse.
    "não pode mergear" CONTAINS "pode mergear" — substring matching is a false
    accept; that trap is test-pinned.
  - fail-closed policy: merge-policy.json maps repo → allowed targets; a repo
    absent from the map cannot be merged at all (Onvida repos will list ONLY
    staging-* targets; main/master stay unreachable by construction).
  - guard GREEN at merge time: the full golden_guard runs on the branch tip in
    a temp worktree BEFORE the merge, and again on the target AFTER it; a
    post-merge RED hard-rolls the target back to the recorded sha.
  - same flock as the delegate (a merge never races a delegate run). The lock
    lives in main() only; run_merge() is the library entry the delegate calls
    while already holding it.
  - audit: every invocation (refusals too) appends to ~/.dasbrowcoder/
    merges.jsonl — verbatim grant message, matched phrase/line, shas, guard
    report hashes. The merge commit message embeds the matched phrase, making
    the authorization permanent git history.
  - NO push code exists in this file, deliberately: a capability that doesn't
    exist can't be granted by accident. Remote pushes are a separate, explicit,
    per-task grant that arrives only with the gated Onvida phase.

Zero LLM. Exit code IS the verdict: 0 = MERGED, 2 = REFUSED/FAIL, 3 = BUSY or
usage error. A terminal verdict is printed even on crash.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

HOME = Path(os.environ.get("DASBROW_HOME", "~/.dasbrowcoder")).expanduser()
LOCK_PATH = HOME / "delegate.lock"
GUARD_PATH = HOME / "golden_guard.py"
POLICY_PATH = HOME / "merge-policy.json"
MERGES_LOG = HOME / "merges.jsonl"
WORKTREES_DIR = HOME / "worktrees"
# Hermes's own message store — the brain cannot write it, so a grant found
# there was really sent by Felipe. Env-overridable for terminal runs only:
# the brain's Bash allowlist matches a command PREFIX, so `VAR=x python …`
# never matches and the brain cannot reach the override.
HERMES_STATE_DB = Path(os.environ.get(
    "DASBROW_HERMES_DB", "~/.hermes/state.db")).expanduser()
PROVENANCE_WINDOW_S = 48 * 3600

# Felipe-approved grant language (REVIEW-r1, 2026-07-15). Case-insensitive,
# own line or message edge; a trailing "?" refuses; negations refuse.
GRANT_PHRASES = ("merge allowed", "pode mergear")
# Negation blocklist, failing closed. "nunca"/"jamais" mirror "never" — a
# missing pt-BR negation would be a false ACCEPT, the only failure that matters.
NEGATION_TOKENS = frozenset({
    "não", "nao", "nunca", "jamais", "not", "never", "don't", "dont", "sem", "no",
})
_EDGE_FOLLOWERS = " .!,;:—–-"  # chars that may follow a start-of-message grant


# ---------- grant parsing (pure, unit-tested) ----------

def _collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _normalize(message: str) -> str:
    norm = unicodedata.normalize("NFC", message)
    return norm.replace("’", "'").replace("‘", "'").casefold()


def _negated(line: str) -> bool:
    tokens = (t.strip(".,!?;:()\"'«»") for t in line.split())
    return any(t in NEGATION_TOKENS for t in tokens)


def _line_grants(line: str, phrase: str) -> bool:
    # A standalone line; "?" deliberately absent — a question is not a grant.
    return line in (phrase, phrase + ".", phrase + "!")


def _edge_grants(whole: str, phrase: str) -> bool:
    if whole == phrase:
        return True
    if whole.startswith(phrase) and whole[len(phrase)] in _EDGE_FOLLOWERS:
        return True
    if whole.endswith(phrase) and whole[-len(phrase) - 1] == " ":
        return True
    return False


def parse_grant(message: str) -> dict | None:
    """Exact-phrase grant extraction. Returns {phrase, line} or None.

    The negation check runs on the LINE containing the match, so a phrase that
    only exists spanning two lines (after collapse) never grants — fail closed.
    """
    if not message:
        return None
    folded = _normalize(message)
    lines = [l for l in (_collapse(l) for l in folded.splitlines()) if l]
    whole = _collapse(folded)
    for phrase in GRANT_PHRASES:
        for line in lines:
            if _line_grants(line, phrase) and not _negated(line):
                return {"phrase": phrase, "line": line}
        if _edge_grants(whole, phrase):
            containing = next((l for l in lines if phrase in l), None)
            if containing is not None and not _negated(containing):
                return {"phrase": phrase, "line": containing}
    return None


def branch_consistency(message: str, branch: str) -> str | None:
    """Any branch-ish reference in the grant message must match --branch.

    Deterministic wrong-branch protection: the brain extracts the branch
    argument, but it cannot merge one branch on a message naming another.
    """
    for m in re.findall(r"dasbrow/[\w./-]+", message):
        if m not in branch:
            return f"grant mentions '{m}' which is not part of '{branch}'"
    for m in re.findall(r"(?<!\d)\d{6}(?!\d)", message):
        if m not in branch:
            return f"grant mentions '{m}' which is not part of '{branch}'"
    return None


def provenance_check(phrase: str, branch: str) -> str | None:
    """Anti-fabrication gate: some RECENT inbound Telegram message in Hermes's
    own store must itself grant this phrase for this branch. parse_grant runs
    on the STORED text — a substring check would re-open the negation trap
    (Felipe's real 'não pode mergear' contains the clean phrase a fabricating
    relay could pass). Skipped only where no store exists (dev checkouts);
    the Pi always has one. Returns a refusal reason or None."""
    if not HERMES_STATE_DB.is_file():
        return None
    import sqlite3  # noqa: PLC0415
    con = sqlite3.connect(f"file:{HERMES_STATE_DB}?mode=ro", uri=True)
    try:
        cutoff = time.time() - PROVENANCE_WINDOW_S
        rows = con.execute(
            "SELECT m.content FROM messages m JOIN sessions s ON s.id = m.session_id "
            "WHERE m.role = 'user' AND s.source = 'telegram' AND m.timestamp >= ? "
            "ORDER BY m.id DESC LIMIT 500", (cutoff,)).fetchall()
    finally:
        con.close()
    for (content,) in rows:
        g = parse_grant(content or "")
        if g and g["phrase"] == phrase and branch_consistency(content, branch) is None:
            return None
    return (f"provenance: no inbound telegram message in the last "
            f"{PROVENANCE_WINDOW_S // 3600}h grants '{phrase}' for this branch "
            f"({HERMES_STATE_DB})")


# ---------- policy (fail-closed) ----------

def policy_for(repo: Path) -> dict | None:
    if not POLICY_PATH.is_file():
        return None
    try:
        repos = json.loads(POLICY_PATH.read_text()).get("repos", {})
    except json.JSONDecodeError:
        return None
    rp = str(Path(repo).expanduser().resolve())
    for key, cfg in repos.items():
        if str(Path(key).expanduser().resolve()) == rp:
            return cfg
    return None


# ---------- git helpers ----------

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def rev(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", ref)


def current_branch(repo: Path) -> str:
    return _git(repo, "branch", "--show-current")


def _ref_exists(repo: Path, ref: str) -> bool:
    return subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=repo, capture_output=True,
    ).returncode == 0


def _is_ancestor(repo: Path, anc: str, desc: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", anc, desc],
        cwd=repo, capture_output=True,
    ).returncode == 0


def _dirty(repo: Path) -> bool:
    return bool(_git(repo, "status", "--porcelain"))


# ---------- guard (subprocess seam; injected in tests) ----------

def run_guard(repo: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(GUARD_PATH), "--repo", str(repo)],
        capture_output=True, text=True, timeout=1800,
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"verdict": "ERROR",
                "reason": f"guard output unparsable: {proc.stdout[-500:]}"}


def _guard_on_branch_tip(repo: Path, branch: str, ts: str, guard_runner) -> dict:
    """Fresh guard on the branch tip in a detached temp worktree — the primary
    checkout is never moved for a verdict that might refuse."""
    wt = WORKTREES_DIR / f"merge-{ts}"
    wt.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "--detach", str(wt), branch)
    try:
        return guard_runner(wt)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                       cwd=repo, capture_output=True)


def _report_digest(report: dict) -> dict:
    return {
        "verdict": report.get("verdict"),
        "sha256": hashlib.sha256(
            json.dumps(report, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "report": report,
    }


# ---------- environment (same launcher pattern as the delegate) ----------

def restore_subscription_env() -> None:
    """The brain's Bash tool strips CLAUDE_CODE_OAUTH_TOKEN from subprocesses.
    This tool is pure git (no credential used), but every launcher the brain
    spawns keeps the one restore pattern so no future addition rediscovers it."""
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


# ---------- orchestration ----------

def _audit(verdict: dict, jsonl: Path | None) -> None:
    """Best-effort audit — must never mask the real verdict."""
    try:
        MERGES_LOG.parent.mkdir(parents=True, exist_ok=True)
        with MERGES_LOG.open("a") as fh:
            fh.write(json.dumps(verdict, default=str) + "\n")
        if jsonl is not None:
            with jsonl.open("a") as fh:
                fh.write(json.dumps({"_type": "MergeVerdict", **verdict},
                                    default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _refused(verdict: dict, reason: str) -> dict:
    verdict.update({"verdict": "REFUSED", "reason": reason})
    return verdict


def run_merge(
    grant_message: str,
    repo: Path,
    branch: str,
    target: str | None = None,
    guard_runner=run_guard,
    jsonl: Path | None = None,
) -> dict:
    ts = time.strftime("%Y%m%d-%H%M%S")
    t0 = time.time()
    verdict: dict = {
        "verdict": "FAIL",
        "reason": "crashed before completion",
        "ts": ts,
        "repo": str(repo),
        "branch": branch,
        "target": target,
        "grant_message": grant_message,
    }
    try:
        grant = parse_grant(grant_message or "")
        if grant is None:
            return _refused(verdict,
                            "no grant: the message does not contain an exact "
                            f"grant phrase {list(GRANT_PHRASES)} on its own line "
                            "or message edge (questions and negations refuse)")
        verdict["grant"] = grant

        mismatch = branch_consistency(grant_message, branch)
        if mismatch is not None:
            return _refused(verdict, f"branch mismatch: {mismatch}")

        prov = provenance_check(grant["phrase"], branch)
        if prov is not None:
            return _refused(verdict, prov)
        verdict["provenance"] = ("verified" if HERMES_STATE_DB.is_file()
                                 else "no store on this machine")

        cfg = policy_for(repo)
        if cfg is None:
            return _refused(verdict,
                            f"repo not in merge policy ({POLICY_PATH}): {repo}")
        targets = cfg.get("targets", [])
        if target is None:
            if not targets:
                return _refused(verdict, "policy lists no targets for this repo")
            target = targets[0]
        if target not in targets:
            return _refused(verdict,
                            f"target '{target}' not allowed by policy {targets}")
        verdict["target"] = target

        if not branch.startswith("dasbrow/"):
            return _refused(verdict,
                            f"only dasbrow/* branches are mergeable, got '{branch}'")
        if not _ref_exists(repo, branch):
            return _refused(verdict, f"branch does not exist: {branch}")
        if not _ref_exists(repo, target):
            return _refused(verdict, f"target does not exist: {target}")
        if rev(repo, branch) == rev(repo, target):
            return _refused(verdict, "nothing to merge: branch is at the target sha")
        if _is_ancestor(repo, branch, target):
            return _refused(verdict, f"already merged: {branch} is an ancestor of {target}")
        if _dirty(repo):
            return _refused(verdict,
                            "working tree not clean — commit/stash before merging")

        pre = _guard_on_branch_tip(repo, branch, ts, guard_runner)
        verdict["guard_pre"] = _report_digest(pre)
        if pre.get("verdict") != "GREEN":
            return _refused(verdict,
                            "guard not GREEN on the branch tip at merge time — "
                            "whatever any earlier run said")

        prior = current_branch(repo)
        before = rev(repo, target)
        verdict["target_sha_before"] = before
        _git(repo, "checkout", "-q", target)
        try:
            _git(repo, "merge", "--no-ff", branch,
                 "-m", f"merge: {branch} — grant: '{grant['phrase']}' (merge tool, {ts})")
        except subprocess.CalledProcessError as exc:
            subprocess.run(["git", "merge", "--abort"], cwd=repo, capture_output=True)
            if prior and prior != target:
                subprocess.run(["git", "checkout", "-q", prior], cwd=repo,
                               capture_output=True)
            verdict.update({"verdict": "FAIL",
                            "reason": "merge failed (conflict?): "
                                      f"{(exc.stderr or exc.stdout or '')[-500:]}"})
            return verdict

        post = guard_runner(repo)
        verdict["guard_post"] = _report_digest(post)
        if post.get("verdict") != "GREEN":
            _git(repo, "reset", "--hard", before)
            verdict.update({"verdict": "FAIL",
                            "reason": "post-merge guard RED — merge rolled back "
                                      f"to {before}"})
            return verdict

        verdict.update({
            "verdict": "MERGED",
            "merge_sha": rev(repo, "HEAD"),
            "reason": "guard GREEN pre and post; merged --no-ff",
        })
        return verdict
    except Exception as exc:  # noqa: BLE001 — verdict line must always exist
        verdict.update({"verdict": "FAIL",
                        "reason": f"{type(exc).__name__}: {exc}"})
        return verdict
    finally:
        verdict["duration_s"] = round(time.time() - t0, 1)
        _audit(verdict, jsonl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--grant-message", required=True,
                    help="Felipe's message VERBATIM — the grant is parsed here, "
                         "never by the model")
    ap.add_argument("--target", help="merge target; default = first policy target")
    ap.add_argument("--jsonl", help="delegate run JSONL to append MergeVerdict to")
    args = ap.parse_args()

    # Fail-closed subscription rule — same contract as delegate and guard.
    if os.environ.get("ANTHROPIC_API_KEY"):
        print(json.dumps({"verdict": "REFUSED",
                          "reason": "ANTHROPIC_API_KEY present — refusing "
                                    "(subscription lane only)"}))
        return 2
    restore_subscription_env()

    repo = Path(args.repo).expanduser()
    if not (repo / ".git").exists():
        print(json.dumps({"verdict": "REFUSED", "reason": f"not a git repo: {repo}"}))
        return 3

    HOME.mkdir(parents=True, exist_ok=True)
    lock_fh = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"verdict": "BUSY",
                          "reason": "a delegate or merge run is in progress "
                                    "(sequential-only)"}))
        return 3

    try:
        verdict = run_merge(
            args.grant_message, repo, args.branch, target=args.target,
            jsonl=Path(args.jsonl).expanduser() if args.jsonl else None,
        )
        print(json.dumps(verdict, indent=2, default=str))
        return 0 if verdict["verdict"] == "MERGED" else 2
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


if __name__ == "__main__":
    sys.exit(main())
