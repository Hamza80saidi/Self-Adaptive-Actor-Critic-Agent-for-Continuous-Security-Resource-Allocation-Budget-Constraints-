"""
stats.py — IQM (interquartile mean) + intervalle de confiance bootstrap
stratifie, methodologie Agarwal et al. 2021 (rliable), reimplementee
manuellement suite a un conflit de dependances de la librairie rliable
(voir notebook.md pour le detail de la deviation documentee).
"""

import numpy as np
from scipy.stats import trim_mean


def iqm(scores: np.ndarray) -> float:
    """Interquartile mean : moyenne apres retrait des 25% valeurs extremes de chaque cote."""
    return float(trim_mean(scores, proportiontocut=0.25))


def stratified_bootstrap_ci(returns_by_seed: list, n_boot: int = 2000, ci: float = 95.0, seed: int = 0):
    """
    Bootstrap stratifie : on rééchantillonne les SEEDS avec remise (pas les
    épisodes individuels directement), puis on regroupe les scores des seeds
    tirées pour calculer l'IQM. C'est ce que fait rliable en pratique.

    Args:
        returns_by_seed: liste de np.ndarray, une entree par seed
        n_boot: nombre de rééchantillonnages bootstrap
        ci: niveau de confiance (95 -> intervalle [2.5, 97.5] percentile)

    Retourne (point_estimate, lower, upper).
    """
    rng = np.random.default_rng(seed)
    n_seeds = len(returns_by_seed)
    boot_iqms = np.zeros(n_boot)

    for b in range(n_boot):
        sampled_idx = rng.integers(0, n_seeds, size=n_seeds)
        pooled = np.concatenate([returns_by_seed[i] for i in sampled_idx])
        boot_iqms[b] = iqm(pooled)

    point = iqm(np.concatenate(returns_by_seed))
    alpha = (100 - ci) / 2
    lower, upper = np.percentile(boot_iqms, [alpha, 100 - alpha])
    return point, float(lower), float(upper)