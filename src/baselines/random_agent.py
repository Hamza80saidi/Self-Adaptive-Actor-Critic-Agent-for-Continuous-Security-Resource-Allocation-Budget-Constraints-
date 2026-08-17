"""
random_agent.py
================
Baseline : allocation aléatoire (logits gaussiens, softmax géré par l'env).
"""

import numpy as np


class RandomAgent:
    def __init__(self, n_resources: int, seed: int = 42):
        self.n_resources = n_resources
        self._rng = np.random.default_rng(seed)

    def select_action(self, obs=None, deterministic: bool = False):
        """Retourne (action, log_prob, value) — même interface que PPOAgent."""
        action = self._rng.normal(size=self.n_resources)
        return action, None, None