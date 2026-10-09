"""Explicit run configuration; experimental values are labeled in YAML/docs."""

from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    """Working defaults from the living specification plus explicit run options."""

    population: int = 100
    neighborhood: int = 20
    neighbor_probability: float = 0.90
    replacement_cap: int = 2
    offspring_policy: str = "variation"  # P2 only; default preserves the current algorithm
    clone_probability: float = 0.20  # provisional CG/CT working value
    replacement_policy: str = "birth"  # P2: birth, permuted, direction
    a7_representative_policy: str = "compression"  # P2: compression or bill_proxy
    crossover_probability: float = 0.90
    mutation_probability: float = 0.20
    mutation_weights: tuple[float, ...] = (0.4, 0.4, 0.2)
    rl_steps: int = 5
    alpha: float = 0.30
    gamma: float = 0.70
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    stagnation_threshold: int = 5
    severity_thresholds: tuple[float, ...] = (0.13, 0.05, 0.60, 0.43, 0.33, 0.16)
    budgets: tuple[int, ...] = (3, 6, 10)
    severity_budget_cutoffs: tuple[float, float] = (1.0, 2.0)
    action_mask_policy: str = "none"
    rl_state_policy: str = "dominant"  # D7: dominant, pooled, compound
    a3_region_policy: str = "load"  # D7: load or tariff; exact acceptance unchanged
    a8_singleton_attempts: int = 2
    a8_member_cap: int = 8
    a8_position_limit: int = 6
    a8_attempt_multiplier: int = 2
    generations: int = 2
    seconds: float | None = None
    exact_evaluation_cap: int | None = None  # E15 equal-exact-evaluation protocol
    seed: int = 1
    method: str = "plain"
    controller: str = "qlearning"
    budget_policy: str = "fixed"
    trigger_delta: float = 0.10  # experimental, not a frozen research parameter
    trigger_quality_gate: bool = True  # E12 ablation only
    trigger_mode: str = "preference"
    fixed_ls_probability: float = 0.5  # test/experimental ablation
    fixed_budget: int = 6
    static_budgets: tuple[int, ...] = (3, 6, 6, 3, 3, 10, 6, 10)
    enabled_operators: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    polish: bool = True
    polish_mode: str = "trajectory"  # E04: none, step, or current trajectory placement
    initialization_attempts: int = 10000
    initialization_perturbation: float = 0.10  # experimental
    random_attempt_multiplier: int = 20  # bounded duplicate generation
    a6_destroy_ratio: float = 0.10  # working value, capped by job count
    instance: str = "examples/smoke.json"
    output: str = "outputs"

    def __post_init__(self) -> None:
        positive_counts = (
            self.population,
            self.neighborhood,
            self.replacement_cap,
            self.generations,
            self.rl_steps,
            self.fixed_budget,
            self.stagnation_threshold,
            self.a8_singleton_attempts,
            self.a8_member_cap,
            self.a8_position_limit,
            self.a8_attempt_multiplier,
            self.initialization_attempts,
            self.random_attempt_multiplier,
            *self.budgets,
            *self.static_budgets,
        )
        if any(type(v) is not int or v < 1 for v in positive_counts):
            raise ValueError("Counts must be positive integers")
        if self.seconds is not None and (not isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError("Wall-clock budget must be finite and positive")
        if self.exact_evaluation_cap is not None and (
            type(self.exact_evaluation_cap) is not int
            or self.exact_evaluation_cap < self.population
        ):
            raise ValueError("Exact evaluation cap must cover initialization")
        if len(self.budgets) != 3 or tuple(sorted(self.budgets)) != self.budgets:
            raise ValueError("Three ordered budget levels required")
        if (
            len(self.severity_budget_cutoffs) != 2
            or any(not isfinite(v) or v <= 0 for v in self.severity_budget_cutoffs)
            or self.severity_budget_cutoffs[0] >= self.severity_budget_cutoffs[1]
        ):
            raise ValueError("Two finite, positive, increasing severity budget cutoffs required")
        if self.action_mask_policy not in ("none", "no_a6", "no_a4a5", "no_a4a5a6"):
            raise ValueError("Unknown experimental action mask policy")
        if self.rl_state_policy not in ("dominant", "pooled", "compound"):
            raise ValueError("Unknown RL state policy")
        if self.a3_region_policy not in ("load", "tariff"):
            raise ValueError("Unknown A3 region policy")
        if self.rl_state_policy != "dominant" and self.action_mask_policy != "none":
            raise ValueError("Historical action masks require the dominant state encoding")
        if self.method != "full" and (
            self.rl_state_policy != "dominant" or self.a3_region_policy != "load"
        ):
            raise ValueError("D7 experimental mechanisms require method=full")
        if self.offspring_policy not in ("variation", "genotype_clone", "phenotype_clone"):
            raise ValueError("Unknown experimental offspring policy")
        if self.replacement_policy not in ("birth", "permuted", "direction"):
            raise ValueError("Unknown experimental replacement policy")
        if self.a7_representative_policy not in ("compression", "bill_proxy"):
            raise ValueError("Unknown experimental A7 representative policy")
        if self.method != "full" and (
            self.offspring_policy != "variation"
            or self.replacement_policy != "birth"
            or self.a7_representative_policy != "compression"
        ):
            raise ValueError("P2 experimental mechanisms require method=full")
        if self.method not in (
            "plain",
            "full",
            "nsga2",
            "nsga2_memetic",
        ) or self.controller not in (
            "qlearning",
            "bandit",
            "random",
        ):
            raise ValueError("Unknown algorithm or controller")
        if self.budget_policy not in (
            "fixed",
            "static",
            "random",
            "severity",
            "coverage",
            "coverage_v2",
            "severity_v2",
            "sequential",
        ):
            raise ValueError("Unknown budget policy")
        if self.trigger_mode not in ("preference", "strict", "always", "fixed"):
            raise ValueError("Unknown trigger mode")
        if self.polish_mode not in ("none", "step", "trajectory"):
            raise ValueError("Unknown Polish placement")
        if not isfinite(self.trigger_delta) or self.trigger_delta < 0:
            raise ValueError("Trigger tolerance must be finite and nonnegative")
        if type(self.trigger_quality_gate) is not bool:
            raise ValueError("Trigger quality gate flag must be bool")
        if (
            len(self.mutation_weights) != 3
            or any(not isfinite(v) or v < 0 for v in self.mutation_weights)
            or sum(self.mutation_weights) <= 0
        ):
            raise ValueError("Three finite nonnegative mutation weights with positive sum required")
        if self.population < 2 or not 2 <= self.neighborhood <= self.population:
            raise ValueError("Require 2 <= neighborhood <= population")
        if self.generations < 1 or self.rl_steps < 1 or self.fixed_budget < 1:
            raise ValueError("Positive generations, trajectory and budget required")
        if len(self.severity_thresholds) != 6 or any(
            not isfinite(t) or t <= 0 for t in self.severity_thresholds
        ):
            raise ValueError("Six positive severity thresholds required")
        if len(self.static_budgets) != 8 or not self.enabled_operators:
            raise ValueError("Eight static budgets and at least one enabled action required")
        if any(a not in range(1, 9) for a in self.enabled_operators):
            raise ValueError("Action outside A1-A8")
        if len(set(self.enabled_operators)) != len(self.enabled_operators):
            raise ValueError("Enabled actions must be unique")
        masked = {
            "none": (),
            "no_a6": (6,),
            "no_a4a5": (4, 5),
            "no_a4a5a6": (4, 5, 6),
        }[self.action_mask_policy]
        if all(a in masked for a in self.enabled_operators):
            raise ValueError("Action mask leaves no action in its target state")
        for value in (
            self.alpha,
            self.gamma,
            self.epsilon_start,
            self.epsilon_end,
            self.crossover_probability,
            self.mutation_probability,
            self.neighbor_probability,
            self.initialization_perturbation,
            self.fixed_ls_probability,
            self.a6_destroy_ratio,
            self.clone_probability,
        ):
            if not 0 <= value <= 1:
                raise ValueError("Probability outside [0,1]")
        if self.a6_destroy_ratio == 0:
            raise ValueError("A6 destroy ratio must be positive")


def load_config(path: str | Path) -> Config:
    """Load explicit YAML overrides; unknown keys are errors."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(Config)}
    if not isinstance(data, dict) or set(data) - known:
        raise ValueError("Unknown config fields")
    for key in (
        "mutation_weights",
        "severity_thresholds",
        "severity_budget_cutoffs",
        "budgets",
        "static_budgets",
        "enabled_operators",
    ):
        if key in data:
            data[key] = tuple(data[key])
    return Config(**data)
