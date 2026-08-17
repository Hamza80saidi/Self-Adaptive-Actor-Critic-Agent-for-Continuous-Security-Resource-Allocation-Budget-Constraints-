"""
analyze_allocation.py — Fig. 5 : allocation moyenne par ressource, avant vs
après entraînement, moyennée sur plusieurs épisodes (plus robuste que
l'observation qualitative de la vidéo, qui ne montre qu'un seul épisode).

Usage: python -m src.utils.analyze_allocation --config configs/config.yaml --checkpoint logs/enhanced/seed_0/best_model.pt --out_dir figures/ --n_episodes 30
"""

import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from src.env import SecurityResourceEnv, RESOURCE_NAMES
from src.agents.actor_critic import PPOAgent
from src.baselines.random_agent import RandomAgent


def mean_allocation_over_episodes(env, agent, n_episodes, deterministic):
    all_allocations = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        for _ in range(env.horizon):
            action, _, _ = agent.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            all_allocations.append(info["allocation"])
    all_allocations = np.array(all_allocations)
    return all_allocations.mean(axis=0), all_allocations.std(axis=0)


def main(config_path, checkpoint, out_dir, n_episodes):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["environment"]
    agent_cfg = cfg["agent"]

    env = SecurityResourceEnv(
        n_resources=env_cfg["n_resources"], horizon=env_cfg["horizon"],
        total_budget=env_cfg["total_budget"], threat_volatility=env_cfg["threat_volatility"],
        reward_lambda=env_cfg["reward_lambda"], seed=env_cfg["seed"],
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    random_agent = RandomAgent(env.n_resources, seed=env_cfg["seed"])
    mean_before, std_before = mean_allocation_over_episodes(env, random_agent, n_episodes, deterministic=False)

    ppo_agent = PPOAgent(obs_dim, action_dim, agent_cfg)
    ppo_agent.load(checkpoint)
    mean_after, std_after = mean_allocation_over_episodes(env, ppo_agent, n_episodes, deterministic=True)

    x = np.arange(len(RESOURCE_NAMES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width/2, mean_before, width, yerr=std_before, capsize=4, label="Before (random)", color="gray")
    ax.bar(x + width/2, mean_after, width, yerr=std_after, capsize=4, label="After (PPO)", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(RESOURCE_NAMES, rotation=20, ha="right")
    ax.set_ylabel("Mean allocation (budget units)")
    ax.set_title(f"Fig. 5 - Mean allocation per resource, before vs after training (n={n_episodes} episodes)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig5_allocation_before_after.pdf"), format="pdf")
    plt.close(fig)
    print("Figure sauvegardee: fig5_allocation_before_after.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="logs/enhanced/seed_0/best_model.pt")
    parser.add_argument("--out_dir", default="figures/")
    parser.add_argument("--n_episodes", type=int, default=30)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.out_dir, args.n_episodes)