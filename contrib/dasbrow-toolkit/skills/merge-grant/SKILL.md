---
name: merge-grant
description: Merge an existing dasbrow/* draft branch when Felipe explicitly grants it ("merge allowed" / "pode mergear"). Use when Felipe replies to a delegate verdict authorizing a merge of work that already sits on a branch.
---

# Merge grant — post-hoc channel

**Default (unchanged, forever):** drafts stay on `dasbrow/*` branches; Felipe merges.
A merge happens ONLY on Felipe's explicit grant, parsed deterministically by the merge
tool — never by you.

**When:** Felipe's own message contains a grant phrase and refers to an existing
`dasbrow/*` branch (usually a reply to an earlier delegate verdict).

**Grant phrases (exact — the tool's law):** `merge allowed` · `pode mergear` —
case-insensitive, on its own line or at the message edge. A trailing `?` is a question,
not a grant. Negations (`não pode mergear`, `not merge allowed`, …) refuse. If Felipe's
wording differs, do NOT translate it into a grant phrase — run the tool with his real
words, relay the refusal, and ask him to send the exact phrase.

**How:** run exactly this (Bash):

```
/home/browbot/.hermes/hermes-agent/venv/bin/python /home/browbot/.dasbrowcoder/merge_branch.py --repo /home/browbot/code/<name> --branch <dasbrow/...> --grant-message "<Felipe's message VERBATIM>"
```

- `--grant-message` is Felipe's message byte-for-byte (escape embedded `"` as `\"`).
  If it can't be passed intact, say so and ask Felipe to resend the grant as a plain
  line — never paraphrase.
- If you are not CERTAIN which branch he means, ask — never guess. The tool refuses a
  grant that names a different branch or timestamp than `--branch`.
- The tool re-runs the full golden guard on the branch tip at merge time and again on
  the merged result; RED refuses or rolls back, regardless of any earlier report. It
  can take a few minutes — use a generous Bash timeout (600000 ms).
- A `BUSY` verdict means a delegate or merge run is live: say so and wait; never kill it.

**Relaying (non-negotiable):** relay the tool's verdict JSON honestly — MERGED (with
`merge_sha`), REFUSED (with the exact reason), FAIL, or BUSY. A REFUSED is a valid
answer: report it, never retry with edited words. Every attempt is audited in
`~/.dasbrowcoder/merges.jsonl` with Felipe's quoted message.

**There is no push.** Local merge only; nothing reaches GitHub. Pushing is a separate
explicit grant that does not exist yet.
