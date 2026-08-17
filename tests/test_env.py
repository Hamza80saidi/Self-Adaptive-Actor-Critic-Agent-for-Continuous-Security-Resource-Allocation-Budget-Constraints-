"""
test_security_env.py
=====================
Tests unitaires critiques pour l'environnement. Lance-les avec :
    pytest tests/test_security_env.py -v

Ces tests sont un bon point à citer dans le rapport pour démontrer la
rigueur de validation de l'environnement.
"""

import numpy as np
import pytest

from src.env import SecurityResourceEnv


def make_env(seed=42):
    return SecurityResourceEnv(
        n_resources=5, horizon=30, total_budget=1000.0,
        threat_volatility=0.3, reward_lambda=0.1, seed=seed,
    )


def test_budget_constraint_respected():
    """La somme de l'allocation doit toujours être égale au budget du jour."""
    env = make_env()
    obs, _ = env.reset(seed=42)
    for _ in range(env.horizon):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        allocation_sum = info["allocation"].sum()
        assert np.isclose(allocation_sum, env.budget_per_day, atol=1e-4), (
            f"Somme de l'allocation ({allocation_sum}) != budget_du_jour ({env.budget_per_day})"
        )
        if terminated or truncated:
            break


def test_episode_length():
    """L'épisode doit se terminer après exactement `horizon` pas de temps."""
    env = make_env()
    obs, _ = env.reset(seed=42)
    n_steps = 0
    for _ in range(env.horizon + 5):  # marge de sécurité
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        n_steps += 1
        if terminated or truncated:
            break
    assert n_steps == env.horizon


def test_observation_bounds():
    """Toutes les observations doivent rester dans observation_space."""
    env = make_env()
    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    for _ in range(env.horizon):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), f"Observation hors bornes: {obs}"
        if terminated or truncated:
            break


def test_reset_reproducibility():
    """Deux reset avec la même seed doivent donner le même état initial."""
    env1 = make_env()
    env2 = make_env()
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    np.testing.assert_array_almost_equal(obs1, obs2)


def test_reward_is_negative_or_zero():
    """Le reward est construit comme -risque - lambda*waste, donc <= 0."""
    env = make_env()
    obs, _ = env.reset(seed=42)
    for _ in range(env.horizon):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward <= 1e-8, f"Reward positif inattendu: {reward}"
        if terminated or truncated:
            break


def test_adaptive_allocation_reduces_risk():
    """
    Vérifie que concentrer le budget sur la ressource la plus efficace contre
    la menace dominante donne un risque plus faible qu'une allocation
    aléatoire -- valide que la fonction de risque a bien un sens économique.
    """
    env = make_env()
    obs, _ = env.reset(seed=42)
    # Forcer une menace dominante connue : Phishing (indice 1)
    env.threat_levels = np.array([0.1, 0.9, 0.1])

    # Action qui concentre tout sur User_Training (indice 2), la ressource
    # la plus efficace contre le phishing (EFFICACY[2,1] = 0.9)
    smart_action = np.array([-5, -5, 5, -5, -5], dtype=np.float32)
    # Action uniforme (logits nuls -> allocation égale entre ressources)
    uniform_action = np.zeros(5, dtype=np.float32)

    env_smart = make_env()
    env_smart.reset(seed=42)
    env_smart.threat_levels = np.array([0.1, 0.9, 0.1])
    _, reward_smart, _, _, _ = env_smart.step(smart_action)

    env_uniform = make_env()
    env_uniform.reset(seed=42)
    env_uniform.threat_levels = np.array([0.1, 0.9, 0.1])
    _, reward_uniform, _, _, _ = env_uniform.step(uniform_action)

    assert reward_smart > reward_uniform, (
        "Une allocation adaptée à la menace dominante devrait donner un "
        "meilleur reward qu'une allocation uniforme."
    )
