"""Print the component-study matrix without dispatching any experiment."""

import argparse
import json

from geo_llm_scheduler.experiments.component_random import planning_summary


def main() -> None:
    """Expose the auditable planning model with explicit concurrency and overhead."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--overhead-seconds", type=float, default=60.0)
    args = parser.parse_args()
    print(json.dumps(planning_summary(args.workers, args.overhead_seconds), indent=2))


if __name__ == "__main__":
    main()
