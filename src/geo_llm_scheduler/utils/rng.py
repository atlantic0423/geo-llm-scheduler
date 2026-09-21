"""Independent reproducible named streams, unaffected by Python hash salt."""

import hashlib
import random


class RNGManager:
    """Derive and retain explicit named streams from a run's master seed."""

    def __init__(self, seed: int):
        self.seed = seed
        self._streams: dict[str, random.Random] = {}

    def stream(self, name: str) -> random.Random:
        """Get a persistent stream; unrelated stream calls cannot disturb it."""
        if name not in self._streams:
            data = f"{self.seed}:{name}".encode()
            self._streams[name] = random.Random(int.from_bytes(hashlib.sha256(data).digest()))
        return self._streams[name]
