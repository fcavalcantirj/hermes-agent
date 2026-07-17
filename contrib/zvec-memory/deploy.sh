#!/usr/bin/env bash
# deploy.sh — copy zvec-memory code to the Pi and verify integrity (sha256
# both sides — house style: a copy nobody verified is a copy nobody made).
# Run from the workstation. Does NOT install the venv, key, or registration —
# it prints those reminders at the end (README.md is the full runbook).
set -euo pipefail

HOST="${ZVEC_DEPLOY_HOST:-browbot@brownet-coder}"
DEST='~/zvec-memory'
FILES=(zvec_memory_core.py zvec_memory_server.py README.md)

cd "$(dirname "$0")"

local_sha() {
  # macOS ships shasum, Linux ships sha256sum — support both sides
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

echo "== deploying to $HOST:$DEST =="
ssh "$HOST" "mkdir -p $DEST"

for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "MISSING LOCAL FILE: $f" >&2; exit 1; }
  scp -q "$f" "$HOST:$DEST/"
done

FAIL=0
for f in "${FILES[@]}"; do
  L="$(local_sha "$f")"
  R="$(ssh "$HOST" "sha256sum $DEST/$f" | awk '{print $1}')"
  if [ "$L" != "$R" ]; then
    echo "SHA MISMATCH: $f  local=$L  remote=$R" >&2
    FAIL=1
  else
    echo "OK  $f  $L"
  fi
done
[ "$FAIL" -eq 0 ] || { echo "DEPLOY FAILED — hashes differ" >&2; exit 1; }

cat <<'REMINDER'

== code deployed & verified. Remaining manual steps (README.md §1,4-8) ==
 1. venv (once):   ssh browbot@brownet-coder \
                     'cd ~/zvec-memory && python3 -m venv venv && venv/bin/pip install zvec fastembed mcp'
 2. warm local model once while online (README §1) — downloads bge-small ONNX
 3. jina key (optional): install -m 600 /dev/null ~/.hermes/zvec-memory/jina.key  (paste key)
 4. register:      claude mcp add -s user zvec-memory -- \
                     /home/browbot/zvec-memory/venv/bin/python /home/browbot/zvec-memory/zvec_memory_server.py
 5. allowlist in ~/.claude/settings.json:
                     mcp__zvec-memory__memory_semantic_search
                     mcp__zvec-memory__memory_index_status
 6. restart the gateway, then run the red-on-demand gates (README §8) BEFORE trusting green
REMINDER
