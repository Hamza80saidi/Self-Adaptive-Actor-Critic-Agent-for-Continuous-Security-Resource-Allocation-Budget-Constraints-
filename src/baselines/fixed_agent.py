"""
fixed_agent.py
==============
Baseline : allocation fixe égale entre ressources (non adaptatif).
"""

import numpy as np


class FixedAgent:
    def __init__(self, n_resources: int):
        self.n_resources = n_resources

    def select_action(self, obs=None, deterministic: bool = False):
        """Retourne (action, log_prob, value) — même interface que PPOAgent."""
        return np.zeros(self.n_resources), None, None