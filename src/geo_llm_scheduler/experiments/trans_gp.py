"""Two-tree GP primitives used by the paper-guided PSLGP-H scheduling adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geo_llm_scheduler.experiments.trans_common import Array

TERMINALS = ("PT", "READY", "REMAIN", "LOAD", "COMPUTE", "VRAM", "ENERGY", "PRICE", "W1", "W2")
FUNCTIONS = ("+", "-", "*", "/", "min", "max")


@dataclass(frozen=True)
class Node:
    """Immutable binary expression; integer tokens address normalized terminals."""

    token: str | int
    children: tuple[Node, ...] = ()

    def evaluate(self, features: Array) -> Array:
        """Vectorized arithmetic with protected division and bounded finite overflow."""
        if isinstance(self.token, int):
            return features[..., self.token]
        a, b = (x.evaluate(features) for x in self.children)
        if self.token == "+":
            result = a + b
        elif self.token == "-":
            result = a - b
        elif self.token == "*":
            result = a * b
        elif self.token == "/":
            result = np.divide(a, b, out=np.ones_like(a), where=b != 0)
        elif self.token == "min":
            result = np.minimum(a, b)
        elif self.token == "max":
            result = np.maximum(a, b)
        else:
            raise ValueError("Unknown GP primitive")
        return np.clip(result, -1e12, 1e12)

    def depth(self) -> int:
        """Return ECJ-style node depth, counting a terminal as one level."""
        return 1 + max((c.depth() for c in self.children), default=0)


Policy = tuple[Node, Node]


def random_tree(rng: np.random.Generator, depth: int, *, full: bool) -> Node:
    """Ramped full/grow; preferences are doubled in the source terminal draw."""
    if depth <= 1 or (not full and rng.random() < 0.5):
        index = int(rng.integers(len(TERMINALS) + 2))
        if index >= len(TERMINALS):
            index -= 2
        return Node(index)
    return Node(
        str(rng.choice(FUNCTIONS)), tuple(random_tree(rng, depth - 1, full=full) for _ in range(2))
    )


def paths(tree: Node) -> list[tuple[int, ...]]:
    """List subtree addresses in deterministic preorder."""
    return [()] + [(i,) + p for i, child in enumerate(tree.children) for p in paths(child)]


def subtree(tree: Node, path: tuple[int, ...]) -> Node:
    """Read the immutable node at an address."""
    return tree if not path else subtree(tree.children[path[0]], path[1:])


def replace(tree: Node, path: tuple[int, ...], other: Node) -> Node:
    """Copy only the ancestry of a replaced subtree."""
    if not path:
        return other
    return Node(
        tree.token,
        tuple(
            replace(c, path[1:], other) if i == path[0] else c for i, c in enumerate(tree.children)
        ),
    )


def node_choice(tree: Node, rng: np.random.Generator) -> tuple[int, ...]:
    """Koza 90% internal-node, 10% terminal selection with available-type fallback."""
    all_paths = paths(tree)
    internal = rng.random() < 0.9
    eligible = [p for p in all_paths if bool(subtree(tree, p).children) == internal] or all_paths
    return eligible[int(rng.integers(len(eligible)))]


def variation(first: Policy, second: Policy, rng: np.random.Generator) -> Policy:
    """Source 80/15/5 crossover/mutation/reproduction and one-try depth-eight limit."""
    probability = rng.random()
    tree_index = int(rng.integers(2))
    result = list(first)
    if probability < 0.8:
        a, b = node_choice(first[tree_index], rng), node_choice(second[tree_index], rng)
        result[tree_index] = replace(first[tree_index], a, subtree(second[tree_index], b))
        # Source AllIndexAllSwap: exchange the entire unselected tree too.
        result[1 - tree_index] = second[1 - tree_index]
    elif probability < 0.95:
        result[tree_index] = replace(
            first[tree_index], node_choice(first[tree_index], rng), random_tree(rng, 4, full=False)
        )
    if result[tree_index].depth() > 8:
        return first
    return result[0], result[1]


def normalize(features: Array) -> Array:
    """Normalize each physical feature across a decision set, preserving preference inputs."""
    result = features.copy()
    raw = result[..., :8]
    low, high = raw.min(axis=-2, keepdims=True), raw.max(axis=-2, keepdims=True)
    result[..., :8] = (raw - low) / np.maximum(high - low, 1e-15)
    return result
