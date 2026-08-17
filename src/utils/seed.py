"""
seed.py
=======
Fixe toutes les graines aléatoires (numpy, torch, random) pour la
reproductibilité des expériences — important à mentionner dans le rapport.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Fixe la graine aléatoire pour numpy, torch et le module random natif."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
