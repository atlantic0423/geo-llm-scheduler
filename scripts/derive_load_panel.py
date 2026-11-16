"""Derive a load-dependent-power copy of a frozen instance panel.

Every job, region and instance stays identical; each input only gains a
``power_model`` section. The output manifest records each source SHA256, so the
derived panel is traceable to the frozen original and never re-generated.
"""

import argparse
import hashlib
import json
from pathlib import Path

from geo_llm_scheduler.domain.models import PowerModel
from geo_llm_scheduler.io.loaders import load_instance


def sha256(path: Path) -> str:
    """Hex SHA256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(source: Path, target: Path, model: PowerModel) -> dict:
    """Write derived inputs and a manifest; refuse to overwrite an existing panel."""
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    (target / "instances").mkdir(parents=True, exist_ok=False)
    power = {
        "beta": model.beta,
        "kappa_prefill": model.kappa_prefill,
        "kappa_decode": model.kappa_decode,
    }
    rows = []
    for row in manifest["instances"]:
        original = source / row["path"]
        if sha256(original) != row["sha256"]:
            raise ValueError(f"Source input changed: {row['path']}")
        data = json.loads(original.read_text(encoding="utf-8"))
        if "power_model" in data:
            raise ValueError(f"Source already has a power model: {row['path']}")
        data["power_model"] = dict(power)
        out = target / row["path"]
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        if load_instance(out).power_model != model:
            raise ValueError(f"Derived input does not load as declared: {row['path']}")
        rows.append(
            {
                **row,
                "sha256": sha256(out),
                "bytes": out.stat().st_size,
                "source_sha256": row["sha256"],
            }
        )
    derived = {
        **{k: v for k, v in manifest.items() if k != "instances"},
        "purpose": manifest["purpose"] + "; load-dependent power copy",
        "power_model": power,
        "derived_from": {"manifest_sha256": sha256(source / "manifest.json")},
        "instances": rows,
    }
    (target / "manifest.json").write_text(
        json.dumps(derived, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return derived


def main() -> None:
    """Command-line entry: source panel, target directory and the power model."""
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--kappa-prefill", type=float, default=1.0)
    p.add_argument("--kappa-decode", type=float, default=0.25)
    a = p.parse_args()
    derived = derive(a.source, a.target, PowerModel(a.beta, a.kappa_prefill, a.kappa_decode))
    print(len(derived["instances"]), sha256(a.target / "manifest.json"))


if __name__ == "__main__":
    main()
