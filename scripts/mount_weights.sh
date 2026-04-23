#!/usr/bin/env bash
# Mount remote safetensors via sshfs so HF transformers can read them without copying locally.
#
# Usage:
#   scripts/mount_weights.sh <remote_host> <remote_path> <local_mount_point>
#
# Example:
#   scripts/mount_weights.sh weights-host@server:/models/gemma-4-26b-a4b-it /mnt/gemma4
#
# Unmount with: fusermount3 -u <local_mount_point>
set -euo pipefail

if [[ $# -lt 3 ]]; then
    cat >&2 <<EOF
Usage: $0 <remote_host_userspec> <remote_path> <local_mount_point>

Args:
  remote_host_userspec  user@host (or just host if SSH config handles user)
  remote_path           absolute path on remote, e.g. /data/weights/gemma-4-26b-a4b-it
  local_mount_point     local directory to mount onto; created if missing

Notes:
  - Read-only mount recommended: we never modify weights.
  - Tune -o reconnect,kernel_cache,auto_cache for stability on long reads.
  - Unmount when done:  fusermount3 -u <local_mount_point>
EOF
    exit 1
fi

REMOTE_HOST="$1"
REMOTE_PATH="$2"
LOCAL_MNT="$3"

mkdir -p "$LOCAL_MNT"

if mountpoint -q "$LOCAL_MNT"; then
    echo "Already mounted: $LOCAL_MNT" >&2
    exit 0
fi

# -o ro                — read-only (safety)
# -o reconnect         — auto-reconnect dropped SSH
# -o kernel_cache      — let kernel cache safely (read-only data)
# -o auto_cache        — invalidate cache on mtime change (irrelevant RO but harmless)
# -o ServerAliveInterval=30 — keepalive over slow/idle links
sshfs \
    -o ro \
    -o reconnect \
    -o kernel_cache \
    -o auto_cache \
    -o ServerAliveInterval=30 \
    "$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_MNT"

echo "Mounted $REMOTE_HOST:$REMOTE_PATH at $LOCAL_MNT (read-only)" >&2
ls -la "$LOCAL_MNT" | head -8 >&2
