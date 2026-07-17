#!/usr/bin/env python3
"""golden_guard.py — deterministic golden-rules gate for delegated coding runs.

Zero LLM. Exit code IS the verdict: 0 = GREEN, 2 = RED, 3 = usage error.
Checks (each independently reported in the JSON verdict on stdout):
  file_ceiling   — no tracked code file exceeds MAX_LINES (markdown exempt)
  inmemory_grep  — "InMemory" must not appear in non-test code
  tests_coverage — `go test ./...` green AND total coverage >= MIN_COVERAGE
                   (zero test files = RED before any coverage math)
  golangci_lint  — `golangci-lint run` exit code (missing binary fails closed)
  govulncheck    — `govulncheck ./...` exit code (missing binary fails closed)
  metered_key    — no ANTHROPIC_API_KEY / sk-ant-api string anywhere in the repo
  token_budget   — (only with --budget-tokens) run-JSONL output tokens within budget
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MAX_LINES = 900
MIN_COVERAGE = 80.0
# Go-installed gate binaries (golangci-lint, govulncheck) live in ~/go/bin, which
# cron/subprocess PATHs usually lack — extend once at import.
os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(Path.home() / "go" / "bin")
CODE_SUFFIXES = {".go", ".py", ".js", ".ts", ".sh", ".c", ".h", ".rs", ".java"}
KEY_PATTERNS = (re.compile(r"ANTHROPIC_API_KEY\s*="), re.compile(r"sk-ant-api\w+"))


def _tracked_files(repo: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    return [repo / line for line in out.splitlines() if line.strip()]


def check_file_ceiling(repo: Path) -> dict:
    offenders = []
    for f in _tracked_files(repo):
        if f.suffix not in CODE_SUFFIXES or not f.is_file():
            continue
        n = sum(1 for _ in f.open(errors="replace"))
        if n > MAX_LINES:
            offenders.append({"file": str(f.relative_to(repo)), "lines": n})
    return {"check": "file_ceiling", "ok": not offenders, "offenders": offenders}


def check_inmemory(repo: Path) -> dict:
    offenders = []
    for f in _tracked_files(repo):
        if f.suffix not in CODE_SUFFIXES or not f.is_file():
            continue
        name = f.name.lower()
        if "test" in name:  # tests may use in-memory doubles
            continue
        for i, line in enumerate(f.open(errors="replace"), 1):
            if "InMemory" in line:
                offenders.append({"file": str(f.relative_to(repo)), "line": i})
    return {"check": "inmemory_grep", "ok": not offenders, "offenders": offenders}


def check_tests_coverage(repo: Path) -> dict:
    # Anti-cheat FIRST: an agent that deletes tests to "pass" the floor must go
    # RED explicitly ("zero tests = fail the build"), before any coverage math.
    if not any(f.name.endswith("_test.go") for f in _tracked_files(repo)):
        return {"check": "tests_coverage", "ok": False, "reason": "zero tests"}
    prof = repo / ".guard-cover.out"
    try:
        test = subprocess.run(
            ["go", "test", "./...", f"-coverprofile={prof}"],
            cwd=repo, capture_output=True, text=True, timeout=600,
        )
        if test.returncode != 0:
            return {"check": "tests_coverage", "ok": False,
                    "reason": "tests failed", "output": test.stdout[-2000:] + test.stderr[-500:]}
        func = subprocess.run(
            ["go", "tool", "cover", f"-func={prof}"],
            cwd=repo, capture_output=True, text=True, timeout=120,
        )
        m = re.search(r"total:\s+\(statements\)\s+([\d.]+)%", func.stdout)
        if not m:
            return {"check": "tests_coverage", "ok": False, "reason": "no total in cover -func"}
        pct = float(m.group(1))
        return {"check": "tests_coverage", "ok": pct >= MIN_COVERAGE,
                "coverage": pct, "floor": MIN_COVERAGE}
    finally:
        prof.unlink(missing_ok=True)


def check_metered_key(repo: Path) -> dict:
    offenders = []
    for f in _tracked_files(repo):
        if not f.is_file():
            continue
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for pat in KEY_PATTERNS:
            if pat.search(text):
                offenders.append({"file": str(f.relative_to(repo)), "pattern": pat.pattern})
    if os.environ.get("ANTHROPIC_API_KEY"):
        offenders.append({"env": "ANTHROPIC_API_KEY present in guard environment"})
    return {"check": "metered_key", "ok": not offenders, "offenders": offenders}


def check_token_budget(jsonl: Path, budget: int) -> dict:
    seen: dict[str, int] = {}
    total = 0
    if not jsonl.is_file():
        return {"check": "token_budget", "ok": False, "reason": f"run jsonl missing: {jsonl}"}
    with jsonl.open() as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = ev.get("message") or {}
            usage = msg.get("usage") or ev.get("usage") or {}
            out = usage.get("output_tokens")
            if not out:
                continue
            mid = msg.get("id") or f"line-{len(seen)}"
            if mid not in seen:  # dedupe: one API message spans multiple JSONL lines
                seen[mid] = out
    total = sum(seen.values())
    return {"check": "token_budget", "ok": total <= budget,
            "output_tokens": total, "budget": budget}


def _run_cli_gate(name: str, cmd: list[str], repo: Path, timeout: int = 600) -> dict:
    """A deterministic external gate: exit code is the verdict; a MISSING tool
    fails closed (reported, never silently skipped)."""
    import shutil as _shutil
    if _shutil.which(cmd[0]) is None:
        return {"check": name, "ok": False, "reason": f"{cmd[0]} not installed"}
    try:
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"check": name, "ok": False, "reason": f"timeout after {timeout}s"}
    return {"check": name, "ok": proc.returncode == 0,
            **({} if proc.returncode == 0 else
               {"output": (proc.stdout + proc.stderr)[-2000:]})}


def check_lint(repo: Path) -> dict:
    return _run_cli_gate("golangci_lint", ["golangci-lint", "run", "--timeout", "300s"], repo)


def check_vuln(repo: Path) -> dict:
    return _run_cli_gate("govulncheck", ["govulncheck", "./..."], repo)


def run_guard(repo: Path, jsonl: Path | None, budget: int | None) -> dict:
    checks = [
        check_file_ceiling(repo),
        check_inmemory(repo),
        check_tests_coverage(repo),
        check_lint(repo),
        check_vuln(repo),
        check_metered_key(repo),
    ]
    if budget is not None:
        checks.append(check_token_budget(jsonl or Path("/nonexistent"), budget))
    verdict = "GREEN" if all(c["ok"] for c in checks) else "RED"
    return {"verdict": verdict, "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--jsonl")
    ap.add_argument("--budget-tokens", type=int)
    args = ap.parse_args()
    repo = Path(args.repo).expanduser()
    if not (repo / ".git").exists():
        print(json.dumps({"verdict": "ERROR", "reason": f"not a git repo: {repo}"}))
        return 3
    report = run_guard(repo, Path(args.jsonl).expanduser() if args.jsonl else None,
                       args.budget_tokens)
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "GREEN" else 2


if __name__ == "__main__":
    sys.exit(main())
