#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Overwrite existing run logs in an older sweep directory with matching runs from a newer sweep.

Only files that already exist in the destination are overwritten (no new runs are created).

Usage:
  overwrite_existing_runs_from_sweep.sh [--dry-run] <OLD_SWEEP_DIR> <NEW_SWEEP_DIR>

Example (your case):
  overwrite_existing_runs_from_sweep.sh --dry-run \
    "Terrarium/experiments/agent_misalignment/outputs/agent_misalignment/20260123-235332 copy" \
    "Terrarium/experiments/agent_misalignment/outputs/agent_misalignment/20260125-123556"

  overwrite_existing_runs_from_sweep.sh \
    "Terrarium/experiments/agent_misalignment/outputs/agent_misalignment/20260123-235332 copy" \
    "Terrarium/experiments/agent_misalignment/outputs/agent_misalignment/20260125-123556"
EOF
}

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi

old_root="${1:-}"
new_root="${2:-}"
if [[ -z "${old_root}" || -z "${new_root}" ]]; then
  usage
  exit 2
fi

old_runs="${old_root%/}/runs/"
new_runs="${new_root%/}/runs/"

if [[ ! -d "${old_runs}" ]]; then
  echo "ERROR: Destination runs dir does not exist: ${old_runs}" >&2
  exit 1
fi
if [[ ! -d "${new_runs}" ]]; then
  echo "ERROR: Source runs dir does not exist: ${new_runs}" >&2
  exit 1
fi

rsync_args=(
  --archive
  --human-readable
  --verbose
  --itemize-changes
  --existing
)
if [[ "${dry_run}" -eq 1 ]]; then
  rsync_args+=(--dry-run)
fi

echo "Syncing (overwrite-only) from:"
echo "  ${new_runs}"
echo "to:"
echo "  ${old_runs}"
echo

rsync "${rsync_args[@]}" "${new_runs}" "${old_runs}"
