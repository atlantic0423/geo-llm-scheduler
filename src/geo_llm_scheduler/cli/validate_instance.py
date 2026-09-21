"""Validate schema, canonical units, phase eligibility and same-region paths."""

import argparse

from geo_llm_scheduler.io.loaders import load_instance


def main() -> None:
    """Validate input without running the search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    args = parser.parse_args()
    p = load_instance(args.instance)
    print(f"Valid: {len(p.jobs)} jobs, {len(p.regions)} regions, {len(p.instances)} instances")


if __name__ == "__main__":
    main()
