"""Single-factor P2 definitions; combinations and server dispatch are out of scope."""

from dataclasses import replace

from geo_llm_scheduler.config import Config

P2_ARMS = ("F6", "CG", "CT", "RPERM", "RDIR", "A7B")


def p2_config(arm: str, base: Config | None = None) -> Config:
    """Build one opt-in arm on a shared full/fixed-6 base, rejecting mixed factors.

    Caller supplies seed, instance and termination budgets via base. Template
    budgets are provisional research inputs, not paper-wide prescribed limits.
    """
    config = base or Config(method="full", fixed_budget=6)
    if (
        config.method != "full"
        or config.budget_policy != "fixed"
        or config.fixed_budget != 6
        or config.offspring_policy != "variation"
        or config.replacement_policy != "birth"
        or config.a7_representative_policy != "compression"
    ):
        raise ValueError("P2 requires an unmixed full/fixed-6 base")
    if arm == "F6":
        return config
    if arm in ("CG", "CT"):
        return replace(
            config, offspring_policy="genotype_clone" if arm == "CG" else "phenotype_clone"
        )
    if arm in ("RPERM", "RDIR"):
        return replace(config, replacement_policy="permuted" if arm == "RPERM" else "direction")
    if arm == "A7B":
        return replace(config, a7_representative_policy="bill_proxy")
    raise ValueError("Unknown P2 arm")
