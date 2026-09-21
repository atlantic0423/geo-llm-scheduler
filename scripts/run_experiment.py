"""Run a paired-seed experiment matrix; each config has its own budget policy."""

import argparse
from pathlib import Path

from geo_llm_scheduler.experiments.runner import run_config

parser = argparse.ArgumentParser()
parser.add_argument("--configs", nargs="+", required=True)
parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
parser.add_argument("--output", default="outputs/matrix")
args = parser.parse_args()
for config in args.configs:
    for seed in args.seeds:
        print(run_config(config, seed, str(Path(args.output) / Path(config).stem / f"seed{seed}")))
