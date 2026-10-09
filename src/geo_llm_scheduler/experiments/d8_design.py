"""D8 experimental switches; stable defaults stay separate from research arms."""

from dataclasses import replace

from geo_llm_scheduler.config import Config

FACTORS = ("A3", "SEQ", "STATE84", "RPERM")
ARMS = tuple(
    "+".join(name for bit, name in enumerate(FACTORS) if mask & (1 << bit)) or "BASE"
    for mask in range(16)
)


def factor_settings(mask: int) -> dict[str, str]:
    """Return exactly the selected factor overrides for a four-bit mask."""
    if not isinstance(mask, int) or isinstance(mask, bool) or mask not in range(16):
        raise ValueError("D8 mask must be an integer from 0 to 15")
    return {
        field: value
        for bit, field, value in (
            (0, "a3_region_policy", "tariff"),
            (1, "budget_policy", "sequential"),
            (2, "rl_state_policy", "compound"),
            (3, "replacement_policy", "permuted"),
        )
        if mask & (1 << bit)
    }


def factor_config(baseline: Config, mask: int) -> Config:
    """Apply the D8 experimental mask to an otherwise unchanged baseline."""
    fields = factor_settings(mask)
    return replace(
        baseline,
        a3_region_policy=fields.get("a3_region_policy", baseline.a3_region_policy),
        budget_policy=fields.get("budget_policy", baseline.budget_policy),
        rl_state_policy=fields.get("rl_state_policy", baseline.rl_state_policy),
        replacement_policy=fields.get("replacement_policy", baseline.replacement_policy),
    )
