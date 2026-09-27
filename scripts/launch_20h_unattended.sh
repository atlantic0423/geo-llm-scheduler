#!/usr/bin/env bash
set -uo pipefail

# A small supervisor for the resumable Python campaign. It retries interrupted
# batches using the same frozen manifest and skips every validated run.
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:?absolute campaign output directory required}"
workers="${2:-16}"
hours="${3:-144}"
python_bin="${GEO_LLM_PYTHON:-python3}"

mkdir -p "$root"
exec 9>"$root/supervisor.lock"
if ! flock -n 9; then
    echo "another campaign supervisor already holds $root/supervisor.lock" >&2
    exit 2
fi
printf '%s\n' "$$" > "$root/supervisor.pid"
trap 'rm -f "$root/supervisor.pid"' EXIT

complete() {
    "$python_bin" - "$root/finished.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
data = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if data.get("complete") and data.get("development") == 63 and data.get("validation") == 300 else 1)
PY
}

attempt=0
while ! complete; do
    attempt=$((attempt + 1))
    echo "$(date -u +%FT%TZ) campaign attempt $attempt, root=$root, workers=$workers"
    PYTHONPATH="$repo/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        "$python_bin" -u "$repo/scripts/run_20h_campaign.py" run \
        --root "$root" --workers "$workers" --hours "$hours"
    result=$?
    if complete; then
        break
    fi
    echo "$(date -u +%FT%TZ) attempt $attempt exited $result without 363/363; resuming in 30 s" >&2
    sleep 30
done
echo "$(date -u +%FT%TZ) complete: 63 development and 300 validation runs"
