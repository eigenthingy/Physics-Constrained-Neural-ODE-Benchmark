"""Run (a shard of) the experiment grid.

    python scripts/run_experiments.py --list                  # size the job
    python scripts/run_experiments.py --shard 0 --n-shards 8  # one worker
    python scripts/run_experiments.py --quick                 # ~5 min sanity run

Shards are disjoint and each appends to its own JSONL, so N workers can run
concurrently with no locking. Re-running a shard skips jobs already recorded.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcnn.experiments import jobs, main  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        js = jobs(a.quick)
        print(f"total jobs: {len(js)}")
        for k, v in sorted(Counter(j["exp"] for j in js).items()):
            print(f"  {k:9s} {v}")
        for k, v in sorted(Counter(j["model"] for j in js).items()):
            print(f"  model {k}: {v}")
        sys.exit(0)

    main(shard=a.shard, n_shards=a.n_shards, out_dir=a.out, quick=a.quick)
