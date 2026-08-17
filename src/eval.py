"""
eval.py — évalue PPO (checkpoint) vs baselines Random/Fixed.
Usage: python -m src.eval --config configs/config.yaml --checkpoint logs/baseline/seed_0/best_model.pt --seed 0 --variant baseline
"""

import argparse
import os
import csv
import yaml
import numpy as np

from src.env import SecurityResourceEnv
from src.agents.actor_critic import PPOAgent
from src.baselines.random_agent import RandomAgent
from src.baselines.fixed_agent import FixedAgent


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def append_csv_row(path, row, fieldnames):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_agent(agent_name, agent, env, n_episodes, seed, variant, csv_path, fieldnames, deterministic=True):
    """Exécute n_episodes et écrit une ligne CSV par épisode."""
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 100000 + ep)
        ep_return = 0.0
        risks = []
        budget_respected = True
        for _ in range(env.horizon):
            action, _, _ = agent.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            risks.append(info["risk_residual"])
            if abs(info["allocation"].sum() - env.budget_per_day) > 1e-3:
                budget_respected = False

        append_csv_row(csv_path, {
            "variant": variant, "seed": seed, "agent": agent_name, "episode": ep,
            "episode_return": ep_return, "mean_risk_residual": float(np.mean(risks)),
            "budget_respected": budget_respected,
        }, fieldnames)
        returns.append(ep_return)

    return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns))}


def evaluate(config: dict, checkpoint: str, seed: int, variant: str = "baseline"):
    env_cfg = config["environment"]
    agent_cfg = config["agent"]
    eval_cfg = config["evaluation"]
    n_episodes = eval_cfg["n_eval_episodes"]

    env = SecurityResourceEnv(
        n_resources=env_cfg["n_resources"], horizon=env_cfg["horizon"],
        total_budget=env_cfg["total_budget"], threat_volatility=env_cfg["threat_volatility"],
        reward_lambda=env_cfg["reward_lambda"], seed=seed,
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    os.makedirs("results", exist_ok=True)
    csv_path = "results/eval_metrics.csv"
    fieldnames = ["variant", "seed", "agent", "episode", "episode_return", "mean_risk_residual", "budget_respected"]

    ppo_agent = PPOAgent(obs_dim, action_dim, agent_cfg)
    ppo_agent.load(checkpoint)
    random_agent = RandomAgent(env.n_resources, seed=seed)
    fixed_agent = FixedAgent(env.n_resources)

    results = {}
    for name, agent, det in [
        (f"ppo_{variant}", ppo_agent, True),
        ("random", random_agent, False),
        ("fixed", fixed_agent, False),
    ]:
        results[name] = evaluate_agent(name, agent, env, n_episodes, seed, variant, csv_path, fieldnames, deterministic=det)
        print(f"[{variant} seed {seed}] {name}: mean_return={results[name]['mean_return']:.2f} "
              f"+/- {results[name]['std_return']:.2f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", type=str, default="baseline", choices=["baseline", "enhanced"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint, args.seed, args.variant)