---
name: coder-delegate
description: Delegate any real coding task (write/change/fix code in a repository) to the coding delegate. Use whenever Felipe asks for code work — never write project code directly in the chat session.
---

# Coder delegate — when and how

**When:** any real coding task — new endpoint, bug fix, refactor, tests. If it changes
a repository, it goes through the delegate. Never write project code in this chat
session; your chat context is for talking to Felipe, not for coding.

**How:** run exactly this (Bash):

```
/home/browbot/.hermes/hermes-agent/venv/bin/python /home/browbot/.dasbrowcoder/delegate_coder.py --task "<one clear, self-contained task>" --repo /home/browbot/code/<name> --grant-message "<Felipe's message VERBATIM>"
```

- `--grant-message` is Felipe's triggering message byte-for-byte (escape embedded `"`
  as `\"`) — ALWAYS pass it; never retype, translate, summarize or add words. The merge
  tool parses it for the exact grant phrases (`merge allowed` / `pode mergear`) itself:
  no phrase → nothing merges, exactly as always. YOU never decide whether a grant
  exists; the tool does. If the message can't be passed intact (quoting breaks), drop
  the flag, say so, and ask Felipe to resend the grant as a plain line.

- Repos live under `~/code/`; the practice repo is `/home/browbot/code/dasbrow-kata`
  (its SPEC.md lists specced-but-unimplemented endpoints — tasks come from there).
- One task per run, sequential-only. A `BUSY` verdict means another run is live: say
  so and wait; never kill it.
- The run can take minutes. Use a generous Bash timeout (600000 ms).

**Relaying the verdict (non-negotiable):**

- Relay the delegate's verdict JSON honestly: verdict, branch, guard result, fix
  rounds, token usage. Label claims [REAL]/[TEST]/[UNVERIFIED] — the guard's GREEN is
  [TEST] evidence, not proof the feature is right.
- On FAIL: report exactly what failed (guard check, branch violation, crash) and
  discuss with Felipe. NEVER silently retry, never "fix" it yourself in chat.
- Default: drafts stay on `dasbrow/*` branches and Felipe merges. The ONLY exception
  is a deterministic grant in Felipe's own words — on PASS the delegate hands his
  verbatim message to the merge tool, which re-runs the full guard at merge time and
  refuses anything less than an exact grant phrase (see the `merge-grant` skill for the
  post-hoc flow). Relay any `merge` block in the verdict unedited — MERGED, REFUSED and
  FAIL are all valid answers; never retry a refusal with edited words. Never touch main
  yourself; there is no push — no push capability exists.
