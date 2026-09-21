"""Run one configured research method and print its artifact summary."""

import argparse
import json

from geo_llm_scheduler.experiments.runner import run_config


def main() -> None:
    """Command-line entry point for baseline/full/bandit and budget ablations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(run_config(args.config, args.seed, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
