"""Aggregate sealed local-search logs without altering any run artifacts."""

import gzip
import json
import os
from collections import Counter
from pathlib import Path

B = Path(os.environ["S4_CAMPAIGN_ROOT"])
R = B / "formal"
D = B / "delivery_v1"
m = json.loads((R / "manifest.json").read_text())
out = []
for key in m["blocks"]:
    for arm in m["arms"][:5]:
        p = R / "runs" / key / arm / "local_search.json.gz"
        with gzip.open(p, "rt") as f:
            d = json.load(f)
        steps = d["steps"]
        polish = d["polish"]
        r = Counter()
        r["steps"] = len(steps)
        r["step_seconds"] = sum(x.get("seconds", 0) for x in steps)
        r["step_archive_entries"] = sum(x.get("entered_archive", 0) for x in steps)
        r["positive_reward"] = sum((x.get("q_reward") or 0) > 0 for x in steps)
        r["negative_reward"] = sum((x.get("q_reward") or 0) < 0 for x in steps)
        r["shift_calls"] = len(polish)
        r["shift_seconds"] = sum(x.get("seconds", 0) for x in polish)
        r["shift_no_proposal"] = sum(x.get("proposals", 0) == 0 for x in polish)
        r["shift_accepted"] = sum(bool(x.get("accepted")) for x in polish)
        r["shift_archive_entries"] = sum(x.get("entered_archive", 0) for x in polish)
        r["shift_hv_positive"] = sum(x.get("hv_gain", 0) > 0 for x in polish)
        r["shift_bill_gain"] = sum(x.get("bill_gain", 0) for x in polish)
        out.append(dict(key=key, arm=arm, **r))
path = D / "ls_stats.json"
path.write_text(json.dumps(out, allow_nan=False))
print(json.dumps(dict(rows=len(out), bytes=path.stat().st_size)))
