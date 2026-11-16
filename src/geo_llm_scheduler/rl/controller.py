"""One shared run-level tabular controller, with Random and Bandit controls."""

import random

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.rl.state import state_count
from geo_llm_scheduler.utils.numeric import EPS_RATIO


class Controller:
    """Maintain configured run-shared values, visits and updates without child resets."""

    def __init__(self, config: Config):
        self.config = config
        size = state_count(config.rl_state_policy)
        self.q = [[0.0] * 8 for _ in range(size)]
        self.visits = [0] * size
        self.updates = [[0] * 8 for _ in range(size)]
        self.selections = [[0] * 8 for _ in range(size)]

    def available_actions(self, state: int) -> tuple[int, ...]:
        """Return enabled actions after the optional, state-specific campaign mask."""
        if state not in range(len(self.q)):
            raise ValueError("State outside the configured state space")
        removed = {
            "none": (),
            "no_a6": (6,),
            "no_a4a5": (4, 5),
            "no_a4a5a6": (4, 5, 6),
        }[self.config.action_mask_policy]
        allowed = tuple(a for a in self.config.enabled_operators if state != 9 or a not in removed)
        if not allowed:
            raise ValueError("Action mask leaves no available action")
        return allowed

    def select(self, state: int, progress: float, rng: random.Random) -> tuple[int, bool]:
        """Epsilon is exploration probability, linearly decayed over run progress."""
        self.visits[state] += 1
        epsilon = self.config.epsilon_start + (
            self.config.epsilon_end - self.config.epsilon_start
        ) * min(1, max(0, progress))
        explore = self.config.controller == "random" or rng.random() < epsilon
        actions = [a - 1 for a in self.available_actions(state)]
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
            0
            if terminal
            else max(self.q[next_state][a - 1] for a in self.available_actions(next_state))
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
