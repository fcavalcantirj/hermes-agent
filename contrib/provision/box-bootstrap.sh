#!/usr/bin/env bash
# box-bootstrap.sh — provision a fresh Debian (12/13) box into a full
# DasBrow-stack Hermes agent: gateway (claude-agent-sdk provider, subscription
# billing, fail-closed) + zvec-memory semantic recall + delegate/guard/merge
# toolkit + skills. Mirrors the proven brownet-coder runbook
# (pis/brownet-coder/provision.md in the casa-viva repo).
#
# Usage (as root on the fresh box):
#   1. cp templates/env.template /root/box.env && chmod 600 /root/box.env
#   2. edit /root/box.env (identity + secrets)
#   3. bash box-bootstrap.sh
#
# Idempotent-ish: safe to re-run after a failed step; existing files are
# overwritten from templates, venvs are reused.
set -euo pipefail

BOX_ENV=/root/box.env
[ -f "$BOX_ENV" ] || { echo "FATAL: $BOX_ENV missing (copy templates/env.template)"; exit 1; }
[ "$(stat -c %a "$BOX_ENV")" = "600" ] || { echo "FATAL: $BOX_ENV must be mode 600"; exit 1; }
set -a; . "$BOX_ENV"; set +a

for v in BOX_USER BOX_NAME BOX_DESC AGENT_NAME BOT_HANDLE OWNER_NAME OWNER_ID \
         TELEGRAM_BOT_TOKEN TELEGRAM_ALLOWED_USERS TELEGRAM_HOME_CHANNEL \
         CLAUDE_CODE_OAUTH_TOKEN; do
  [ -n "${!v:-}" ] || { echo "FATAL: $v is empty in $BOX_ENV"; exit 1; }
done
REPO_URL="${REPO_URL:-https://github.com/fcavalcantirj/hermes-agent.git}"
STACK_BRANCH="${STACK_BRANCH:-dasbrow/stack}"
GO_VERSION="${GO_VERSION:-1.23.4}"
TZ="${TZ:-America/Sao_Paulo}"

HERE="$(cd "$(dirname "$0")" && pwd)"          # contrib/provision
CONTRIB="$(dirname "$HERE")"                    # contrib/
HOMEDIR="/home/$BOX_USER"

say() { echo; echo "== $* =="; }

# ── 1. system ────────────────────────────────────────────────────────
say "system packages"
timedatectl set-timezone "$TZ" || true
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3 python3-venv python3-pip curl ca-certificates \
  build-essential nodejs npm jq rsync >/dev/null

ARCH=$(dpkg --print-architecture)   # arm64 | amd64
if ! command -v /usr/local/go/bin/go >/dev/null 2>&1; then
  say "go $GO_VERSION ($ARCH)"
  curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-${ARCH}.tar.gz" | tar -C /usr/local -xz
fi
ln -sf /usr/local/go/bin/go /usr/local/bin/go

# ── 2. user ──────────────────────────────────────────────────────────
say "user $BOX_USER + linger"
id "$BOX_USER" >/dev/null 2>&1 || useradd -m -s /bin/bash "$BOX_USER"
loginctl enable-linger "$BOX_USER"
BOX_UID=$(id -u "$BOX_USER")
RUNDIR="/run/user/$BOX_UID"
# user manager needs a moment after enable-linger on first boot
for _ in $(seq 1 10); do [ -d "$RUNDIR" ] && break; sleep 1; done

as_user() { runuser -u "$BOX_USER" -- bash -lc "$*"; }
as_user_sysd() { runuser -u "$BOX_USER" -- env "XDG_RUNTIME_DIR=$RUNDIR" \
  "DBUS_SESSION_BUS_ADDRESS=unix:path=$RUNDIR/bus" bash -lc "$*"; }

# ── 3. hermes fork ───────────────────────────────────────────────────
say "hermes fork @ $STACK_BRANCH"
as_user "mkdir -p ~/.hermes && cd ~/.hermes && \
  { [ -d hermes-agent/.git ] || git clone -q '$REPO_URL' hermes-agent; } && \
  cd hermes-agent && git fetch -q origin && git checkout -q '$STACK_BRANCH' && \
  git pull -q --ff-only origin '$STACK_BRANCH' || true"
as_user "cd ~/.hermes/hermes-agent && { [ -d venv ] || python3 -m venv venv; } && \
  venv/bin/pip install -q -e '.[claude-agent-sdk]' pytest"
as_user "cd ~/.hermes/hermes-agent && { [ ! -f package.json ] || npm install --omit=dev --silent; }"

say "fork smoke suites (claude_sdk_runtime + providers)"
# inner shell needs its own pipefail — bash -lc does NOT inherit ours, and
# without it `| tail -1` eats a failing suite (fail-open, proven on first run)
as_user "set -o pipefail; cd ~/.hermes/hermes-agent && venv/bin/python -m pytest \
  tests/agent/test_claude_sdk_runtime.py tests/providers/ -q 2>&1 | tail -1"

# ── 4. identity + config from templates ──────────────────────────────
say "identity + config"
render() {  # render <template> <dest>  (owner-substituted)
  sed -e "s|__HOME__|$HOMEDIR|g" -e "s|__AGENT_NAME__|$AGENT_NAME|g" \
      -e "s|__BOT_HANDLE__|$BOT_HANDLE|g" -e "s|__OWNER_NAME__|$OWNER_NAME|g" \
      -e "s|__OWNER_ID__|$OWNER_ID|g" -e "s|__BOX_NAME__|$BOX_NAME|g" \
      -e "s|__BOX_DESC__|$BOX_DESC|g" -e "s|__TZ__|$TZ|g" "$1" > "$2"
}
as_user "mkdir -p ~/.dasbrowcoder ~/.claude/skills ~/.hermes/memories ~/code"

render "$HERE/templates/SOUL.template.md"  /tmp/SOUL.md
render "$HERE/templates/USER.template.md"  /tmp/USER.md
render "$HERE/templates/settings.json.template" /tmp/settings.json
install -o "$BOX_USER" -g "$BOX_USER" -m 644 /tmp/SOUL.md "$HOMEDIR/.dasbrowcoder/SOUL.md"
install -o "$BOX_USER" -g "$BOX_USER" -m 644 /tmp/USER.md "$HOMEDIR/.hermes/memories/USER.md"
install -o "$BOX_USER" -g "$BOX_USER" -m 644 /tmp/settings.json "$HOMEDIR/.claude/settings.json"
rm -f /tmp/SOUL.md /tmp/USER.md /tmp/settings.json
as_user "[ -f ~/.hermes/memories/MEMORY.md ] || echo '# Working memory (self-curated via the memory tool)' > ~/.hermes/memories/MEMORY.md"
install -o "$BOX_USER" -g "$BOX_USER" -m 644 "$HERE/templates/config.yaml" "$HOMEDIR/.hermes/config.yaml"

say "secrets (~/.hermes/.env, mode 600)"
cat > /tmp/hermes.env <<EOF
CLAUDE_CODE_OAUTH_TOKEN=$CLAUDE_CODE_OAUTH_TOKEN
TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
TELEGRAM_ALLOWED_USERS=$TELEGRAM_ALLOWED_USERS
TELEGRAM_HOME_CHANNEL=$TELEGRAM_HOME_CHANNEL
TELEGRAM_HOME_CHANNEL_NAME="${TELEGRAM_HOME_CHANNEL_NAME:-Owner DM}"
HERMES_CLAUDE_SDK_APPEND_FILE=$HOMEDIR/.dasbrowcoder/SOUL.md
EOF
install -o "$BOX_USER" -g "$BOX_USER" -m 600 /tmp/hermes.env "$HOMEDIR/.hermes/.env"
rm -f /tmp/hermes.env

# ── 5. toolkit + skills ──────────────────────────────────────────────
say "dasbrow toolkit + skills"
for f in delegate_coder.py golden_guard.py merge_branch.py; do
  install -o "$BOX_USER" -g "$BOX_USER" -m 644 "$CONTRIB/dasbrow-toolkit/$f" "$HOMEDIR/.dasbrowcoder/$f"
done
install -o "$BOX_USER" -g "$BOX_USER" -m 644 "$HERE/templates/merge-policy.json" "$HOMEDIR/.dasbrowcoder/merge-policy.json"
as_user "mkdir -p ~/.dasbrowcoder/agents"
install -o "$BOX_USER" -g "$BOX_USER" -m 644 "$CONTRIB"/dasbrow-toolkit/agents/*.md "$HOMEDIR/.dasbrowcoder/agents/"
# skills live in BOTH places (brain loads ~/.claude/skills — proven gotcha).
# Copy from the USER'S OWN clone (same branch) — /root/stack is unreadable to
# the box user (proven on first run: rsync exit 23, Permission denied).
as_user "rsync -a ~/.hermes/hermes-agent/contrib/dasbrow-toolkit/skills/ ~/.claude/skills/ && \
         rsync -a ~/.hermes/hermes-agent/contrib/dasbrow-toolkit/skills/ ~/.dasbrowcoder/skills/"

# ── 6. zvec-memory sidecar ───────────────────────────────────────────
say "zvec-memory sidecar"
as_user "rsync -a --exclude __pycache__ --exclude .pytest_cache ~/.hermes/hermes-agent/contrib/zvec-memory/ ~/zvec-memory/"
as_user "cd ~/zvec-memory && { [ -d venv ] || python3 -m venv venv; } && \
  venv/bin/pip install -q zvec fastembed mcp pytest"
say "zvec test suite on this box"
as_user "set -o pipefail; cd ~/zvec-memory && venv/bin/python -m pytest tests/ -q 2>&1 | tail -1"
say "warm local embedder (downloads bge-small once)"
as_user "set -o pipefail; cd ~/zvec-memory && venv/bin/python -c 'import zvec_memory_core as c; print(\"warm:\", len(c.embed_local([\"warmup\"])[0]))' 2>&1 | tail -1"
as_user "mkdir -p ~/.hermes/zvec-memory"
if [ -n "${JINA_API_KEY:-}" ]; then
  printf '%s' "$JINA_API_KEY" > /tmp/jina.key
  install -o "$BOX_USER" -g "$BOX_USER" -m 600 /tmp/jina.key "$HOMEDIR/.hermes/zvec-memory/jina.key"
  rm -f /tmp/jina.key
  echo "jina key installed (quality lane ON)"
else
  echo "no jina key — local-only lane (fully functional)"
fi
say "register zvec-memory MCP server (user scope)"
as_user "python3 - <<'PYEOF'
import json, os
p = os.path.expanduser('~/.claude.json')
d = json.load(open(p)) if os.path.exists(p) else {}
d.setdefault('mcpServers', {})['zvec-memory'] = {
    'type': 'stdio',
    'command': os.path.expanduser('~/zvec-memory/venv/bin/python'),
    'args': [os.path.expanduser('~/zvec-memory/zvec_memory_server.py')],
    'env': {},
}
json.dump(d, open(p, 'w'), indent=2)
print('registered:', list(d['mcpServers']))
PYEOF"

# ── 7. optional extras (best-effort, never fatal) ────────────────────
say "claude CLI (native, best-effort)"
as_user "command -v claude >/dev/null 2>&1 || curl -fsSL https://claude.ai/install.sh | bash" \
  || echo "WARN: claude CLI install failed (not required — SDK bundles its own)"
say "go tools for the guard (best-effort, slow)"
# cd ~ first — runuser inherits the script's cwd (/root/stack), unreadable to
# the box user, and go's toolchain refuses an unreadable cwd (proven on run 2)
as_user "cd ~ && export PATH=\$PATH:/usr/local/go/bin:\$HOME/go/bin && \
  go install golang.org/x/tools/gopls@latest && \
  go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest && \
  go install golang.org/x/vuln/cmd/govulncheck@latest" \
  || echo "WARN: go tools incomplete — guard lint/vuln steps will fail until installed"

# ── 8. gateway unit ──────────────────────────────────────────────────
say "systemd user unit + streaming drop-in"
as_user "mkdir -p ~/.config/systemd/user/hermes-gateway.service.d"
render "$HERE/templates/hermes-gateway.service.template" /tmp/hermes-gateway.service
install -o "$BOX_USER" -g "$BOX_USER" -m 644 /tmp/hermes-gateway.service \
  "$HOMEDIR/.config/systemd/user/hermes-gateway.service"
rm -f /tmp/hermes-gateway.service
install -o "$BOX_USER" -g "$BOX_USER" -m 644 "$HERE/templates/streaming.conf" \
  "$HOMEDIR/.config/systemd/user/hermes-gateway.service.d/streaming.conf"
as_user_sysd "systemctl --user daemon-reload && systemctl --user enable --now hermes-gateway"
sleep 6
as_user_sysd "systemctl --user is-active hermes-gateway"

# ── 9. report ────────────────────────────────────────────────────────
say "DONE — $AGENT_NAME on $BOX_NAME"
cat <<EOF
Gateway:   $(as_user_sysd 'systemctl --user is-active hermes-gateway')
Owner:     $OWNER_NAME ($OWNER_ID)  allowlist: $TELEGRAM_ALLOWED_USERS
Bot:       @$BOT_HANDLE
Next (from the operator's machine):
  1. red-on-demand gates: zvec JSON-RPC probe + missing-DB + StoreBusy
     (contrib/zvec-memory/README.md §8), headless PONG via claude -p
  2. owner sends /start to @$BOT_HANDLE, then a PONG message
  3. create the nightly consolidation cron once the gateway answers:
     hermes cron create '0 3 * * *' '<consolidation prompt>' \\
       --name nightly-memory-consolidation --deliver telegram
EOF
