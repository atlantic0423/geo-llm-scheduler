"""Run one configured research method and print its artifact summary."""

import argparse
import json

from geo_llm_scheduler.default_algorithm import DEFAULT_PROFILE, run_default_config
from geo_llm_scheduler.experiments.runner import run_config


def main() -> None:
    """Run the adopted working algorithm, or an explicitly selected historical config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Explicit profile or historical experiment YAML")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument("--instance")
    parser.add_argument("--calibration", help="Frozen S4 calibration JSON; required for default")
    parser.add_argument("--calibration-key")
    parser.add_argument("--seconds", type=float, help="Override the inclusive size-based budget")
    args = parser.parse_args()
    values = vars(args)
    config = values.pop("config")
    try:
        result = (
            run_config(config, **values)
            if config
            else run_default_config({"profile": DEFAULT_PROFILE}, **values)
        )
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
