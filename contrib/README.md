# contrib/ — the DasBrow stack (fork-only distribution layer)

This directory exists ONLY on the `dasbrow/stack` branch — never on
`feat/claude-agent-sdk-provider` (the upstream-facing branch the public PRs are
cut from) and never in upstream PRs. It packages everything a production box
runs beyond hermes core, so "a fresh box like mine" is one bootstrap run.

**Coupling contract: nothing in `contrib/` is imported by hermes core, ever.**
zvec-memory attaches via user-scope MCP registration (`~/.claude.json`); the
toolkit is invoked by the agent through its Bash allowlist. Delete `contrib/`
and hermes is untouched.

## Contents

- `zvec-memory/` — semantic-recall MCP sidecar (dual-vector: jina API quality
  lane + bge-small local lane, FTS stays the keyword floor). Verbatim from the
  proven deployment (casa-viva `pis/brownet-coder/dasbrowcoder/zvec-memory/`,
  gates passed 2026-07-17); its README is the runbook, SPEC.md the contract.
- `dasbrow-toolkit/` — the delegate → golden-guard → merge-grant pipeline
  (`delegate_coder.py`, `golden_guard.py`, `merge_branch.py`) + skills
  (golden-rules, coder-delegate, merge-grant) + reviewer agent. Verbatim from
  the proven brownet-coder deployment; owner-name strings in comments/help are
  historical (the author), the code is owner-generic — grants verify against
  the local box's own state.db.
- `provision/` — `box-bootstrap.sh` + templates: one script takes a fresh
  Debian (12/13, arm64 or amd64) box to a running agent. Parameterized via
  `/root/box.env` (identity + secrets — see `templates/env.template`; never
  commit a filled copy).

## Provisioning a new box

```bash
# on the fresh box, as root:
git clone <this fork> /root/stack && cd /root/stack && git checkout dasbrow/stack
cp contrib/provision/templates/env.template /root/box.env && chmod 600 /root/box.env
# fill /root/box.env, then:
bash contrib/provision/box-bootstrap.sh
```

After it reports the gateway active: run the red-on-demand gates
(`contrib/zvec-memory/README.md` §8 + a headless PONG), have the owner
`/start` the bot, then create the nightly consolidation cron.

## Maintenance

`dasbrow/stack` = `feat/claude-agent-sdk-provider` + this directory. When the
feature branch advances, rebase: `git rebase feat/claude-agent-sdk-provider`
— contrib never conflicts with core. Boxes update with `git pull --ff-only`.
