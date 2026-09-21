"""One shared run-level tabular controller, with Random and Bandit controls."""

import random

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.utils.numeric import EPS_RATIO


class Controller:
    """Maintain 42x8 values, state visits and action updates without per-child resets."""

    def __init__(self, config: Config):
        self.config = config
        self.q = [[0.0] * 8 for _ in range(42)]
        self.visits = [0] * 42
        self.updates = [[0] * 8 for _ in range(42)]
        self.selections = [[0] * 8 for _ in range(42)]

    def select(self, state: int, progress: float, rng: random.Random) -> tuple[int, bool]:
        """Epsilon is exploration probability, linearly decayed over run progress."""
        self.visits[state] += 1
        epsilon = self.config.epsilon_start + (
            self.config.epsilon_end - self.config.epsilon_start
        ) * min(1, max(0, progress))
        explore = self.config.controller == "random" or rng.random() < epsilon
        actions = [a - 1 for a in self.config.enabled_operators]
        if explore:
            chosen = rng.choice(actions)
        else:
            best = max(self.q[state][a] for a in actions)
            chosen = rng.choice([a for a in actions if self.q[state][a] == best])
        self.selections[state][chosen] += 1
        return chosen + 1, explore

    def update(
        self, state: int, action: int, reward: float, next_state: int, terminal: bool = False
    ) -> None:
        """Apply a Q update; Bandit sets gamma=0, Random does not learn."""
        if self.config.controller == "random":
            return
        gamma = 0 if self.config.controller == "bandit" else self.config.gamma
        future = (
            0 if terminal else max(self.q[next_state][a - 1] for a in self.config.enabled_operators)
        )
        old = self.q[state][action - 1]
        self.q[state][action - 1] = old + self.config.alpha * (reward + gamma * future - old)
        self.updates[state][action - 1] += 1


def reward(before: float, after: float, feasible: int) -> float:
    """No feasible candidate: -1; no improvement: 0; otherwise clipped percentage."""
    if feasible == 0:
        return -1.0
    delta = (before - after) / (before + EPS_RATIO)
    return min(10.0, 100 * delta) if delta > 0 else 0.0
