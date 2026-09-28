#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:?absolute E15 campaign root required}"
python_bin="${GEO_LLM_PYTHON:-python3}"
mkdir -p "$root/logs"
workers="$($python_bin - "$root/resource_plan.json" <<'PY'
import json, sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
print(plan['worker_count'])
PY
)"
if [[ -f "$root/stop.request" ]]; then
    echo "stop.request exists; remove only after reviewing campaign state" >&2
    exit 2
fi
nohup setsid env PYTHONPATH="$repo/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    "$python_bin" -u "$repo/scripts/run_e15_campaign.py" watchdog \
    --root "$root" --workers "$workers" \
    > "$root/logs/watchdog.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$root/watchdog.launch.pid"
echo "watchdog_pid=$pid workers=$workers root=$root"
