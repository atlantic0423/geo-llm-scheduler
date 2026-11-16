"""NumPy port of CMOEA-AOP's small DDPG, with explicit RNG and Adam.

Topology, replay, OU noise, target updates and defaults follow upstream DDPG.m.
Float64 replaces MATLAB single arrays. No pretrained network or global RNG.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from geo_llm_scheduler.experiments.trans_common import Array


class Network:
    """Two ReLU layers and a linear or tanh head, with explicit analytic gradients."""

    def __init__(self, sizes: tuple[int, ...], rng: np.random.Generator, tanh: bool) -> None:
        self.tanh = tanh
        self.weights: list[Array] = [
            rng.uniform(-np.sqrt(6 / (a + b)), np.sqrt(6 / (a + b)), (a, b))
            for a, b in zip(sizes[:-1], sizes[1:], strict=True)
        ]
        self.biases: list[Array] = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.weights + self.biases]
        self.v = [np.zeros_like(p) for p in self.weights + self.biases]

    def forward(self, x: Array) -> tuple[Array, list[Array]]:
        """Return outputs and layer activations for an explicit backward pass."""
        activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = x @ w + b
            x = np.maximum(x, 0) if i < len(self.weights) - 1 else np.tanh(x) if self.tanh else x
            activations.append(x)
        return x, activations

    def backward(self, gradient: Array, activations: list[Array]) -> tuple[list[Array], Array]:
        """Differentiate a caller-supplied loss derivative through all parameters."""
        gw: list[Array] = [np.zeros_like(w) for w in self.weights]
        gb: list[Array] = [np.zeros_like(b) for b in self.biases]
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1 and self.tanh:
                gradient = gradient * (1 - activations[i + 1] ** 2)
            elif i < len(self.weights) - 1:
                gradient = gradient * (activations[i + 1] > 0)
            gw[i] = activations[i].T @ gradient
            gb[i] = gradient.sum(axis=0)
            gradient = gradient @ self.weights[i].T
        return gw + gb, gradient

    def adam(self, gradients: list[Array], iteration: int, rate: float) -> None:
        """Apply Adam with upstream framework defaults and bias correction."""
        for i, (parameter, grad) in enumerate(
            zip(self.weights + self.biases, gradients, strict=True)
        ):
            if not np.all(np.isfinite(grad)):
                raise FloatingPointError("Non-finite DDPG gradient")
            self.m[i] = 0.9 * self.m[i] + 0.1 * grad
            self.v[i] = 0.999 * self.v[i] + 0.001 * grad**2
            parameter -= (
                rate
                * (self.m[i] / (1 - 0.9**iteration))
                / (np.sqrt(self.v[i] / (1 - 0.999**iteration)) + 1e-8)
            )

    def blend(self, online: Network, tau: float) -> None:
        """Soft-update target parameters without changing optimizer state."""
        for target, source in zip(
            self.weights + self.biases, online.weights + online.biases, strict=True
        ):
            target[:] = tau * source + (1 - tau) * target


class DDPG:
    """Original AOP 6-state/3-action controller and 500-transition replay buffer."""

    def __init__(self, rng: np.random.Generator, states: int = 6, actions: int = 3) -> None:
        self.rng = rng
        self.bound = 5.0
        self.actor = Network((states, 32, 32, actions), rng, True)
        self.critic = Network((states + actions, 32, 32, 1), rng, False)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.buffer: list[tuple[Array, Array, float, Array, bool]] = []
        self.pointer = 0
        self.iteration = 0
        self.noise = np.zeros(actions)
        self.noise_scale = 0.3

    def select(self, state: Array, *, explore: bool = True) -> Array:
        """Return clipped actor output with persistent Ornstein-Uhlenbeck noise."""
        action = self.actor.forward(state.reshape(1, -1))[0][0] * self.bound
        if explore:
            self.noise += -0.15 * self.noise + 0.2 * self.rng.normal(size=self.noise.shape)
            action = np.clip(action + self.noise_scale * self.noise, -self.bound, self.bound)
        return action

    def store(
        self, state: Array, action: Array, reward: float, next_state: Array, done: bool
    ) -> None:
        """Store a copied transition, overwriting in source ring-buffer order."""
        row = (state.copy(), action.copy(), reward, next_state.copy(), done)
        if len(self.buffer) < 500:
            self.buffer.append(row)
        else:
            self.buffer[self.pointer] = row
        self.pointer = (self.pointer + 1) % 500

    def train(self) -> tuple[float, float]:
        """One replay minibatch, critic/actor Adam updates and soft target update."""
        if len(self.buffer) < 32:
            return 0.0, 0.0
        batch = [self.buffer[int(i)] for i in self.rng.integers(len(self.buffer), size=32)]
        states = np.asarray([b[0] for b in batch])
        actions = np.asarray([b[1] for b in batch])
        rewards = np.asarray([b[2] for b in batch])[:, None]
        next_states = np.asarray([b[3] for b in batch])
        done = np.asarray([b[4] for b in batch])[:, None]
        next_actions = self.actor_target.forward(next_states)[0] * self.bound
        future = self.critic_target.forward(np.column_stack((next_states, next_actions)))[0]
        target = rewards + 0.99 * (1 - done) * future
        predicted, cache = self.critic.forward(np.column_stack((states, actions)))
        critic_loss = float(np.mean((predicted - target) ** 2))
        gradients, _ = self.critic.backward(2 * (predicted - target) / 32, cache)
        self.iteration += 1
        self.critic.adam(gradients, self.iteration, 1e-3)
        actor_values, acache = self.actor.forward(states)
        qvalues, ccache = self.critic.forward(np.column_stack((states, actor_values * self.bound)))
        _, inputs = self.critic.backward(-np.ones_like(qvalues) / 32, ccache)
        gradients, _ = self.actor.backward(inputs[:, states.shape[1] :] * self.bound, acache)
        self.actor.adam(gradients, self.iteration, 1e-4)
        self.actor_target.blend(self.actor, 0.005)
        self.critic_target.blend(self.critic, 0.005)
        self.noise_scale *= 0.9995
        return critic_loss, -float(qvalues.mean())
