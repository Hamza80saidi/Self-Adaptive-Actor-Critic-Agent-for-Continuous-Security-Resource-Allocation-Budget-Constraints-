"""
train.py — boucle d'entraînement PPO multi-seed.
Usage: python -m src.train --config configs/config.yaml --seed 0
"""

import argparse
import os
import csv
import yaml
import numpy as np
import torch

from src.env import SecurityResourceEnv
from src.agents.actor_critic import PPOAgent
from src.utils.seed import set_seed
from src.utils.logger import Logger


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def collect_episode(env, agent):
    obs, _ = env.reset()
    data = {k: [] for k in ["obs", "actions", "log_probs", "values", "rewards", "dones", "risks", "wastes"]}
    for _ in range(env.horizon):
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        data["obs"].append(obs)
        data["actions"].append(action)
        data["log_probs"].append(log_prob)
        data["values"].append(value)
        data["rewards"].append(reward)
        data["dones"].append(float(terminated or truncated))
        data["risks"].append(info["risk_residual"])
        data["wastes"].append(info["waste"])
        obs = next_obs
    return data


def append_csv_row(path, row, fieldnames):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def entropy_schedule(episode, n_episodes, start, end):
    """Décroissance linéaire de l'entropy_coef (variant='enhanced' uniquement)."""
    progress = min(episode / max(n_episodes - 1, 1), 1.0)
    return start + progress * (end - start)


def train(config: dict, seed: int, variant: str = "baseline", use_annealing: bool = None):
    """
    variant : tag utilisé pour les chemins de logs/CSV (peut être n'importe
              quelle chaîne, ex: "ablation_aggressive_decay").
    use_annealing : force l'activation de l'entropy annealing. Si None,
              déduit de variant == "enhanced" (comportement par défaut
              pour train.py appelé en ligne de commande).
    """
    env_cfg = config["environment"]
    agent_cfg = config["agent"]
    train_cfg = config["training"]
    enh_cfg = config.get("enhancement", {})
    if use_annealing is None:
        use_annealing = (variant == "enhanced")

    set_seed(seed)
    env = SecurityResourceEnv(
        n_resources=env_cfg["n_resources"], horizon=env_cfg["horizon"],
        total_budget=env_cfg["total_budget"], threat_volatility=env_cfg["threat_volatility"],
        reward_lambda=env_cfg["reward_lambda"], seed=seed,
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent_cfg = {**agent_cfg, "batch_size": train_cfg["batch_size"], "n_epochs_per_update": train_cfg["n_epochs_per_update"]}
    agent = PPOAgent(obs_dim, action_dim, agent_cfg)

    log_dir = os.path.join(train_cfg["log_dir"], variant, f"seed_{seed}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    logger = Logger(log_dir)
    csv_path = "results/train_metrics.csv"
    fieldnames = ["variant", "seed", "episode", "step", "episode_return", "mean_risk_residual",
                  "mean_waste", "policy_loss", "value_loss", "entropy", "entropy_coef"]

    buffer_episodes = []
    global_step = 0
    best_return = -np.inf

    for episode in range(train_cfg["n_episodes"]):
        if use_annealing:
            agent.entropy_coef = entropy_schedule(
                episode, train_cfg["n_episodes"],
                enh_cfg["entropy_coef_start"], enh_cfg["entropy_coef_end"],
            )
        # sinon agent.entropy_coef reste celui de agent_cfg (fixe, comportement baseline)

        ep_data = collect_episode(env, agent)
        global_step += len(ep_data["rewards"])

        rewards = np.array(ep_data["rewards"], dtype=np.float32)
        values = np.array(ep_data["values"], dtype=np.float32)
        dones = np.array(ep_data["dones"], dtype=np.float32)
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value=0.0)
        ep_data["advantages"] = advantages
        ep_data["returns"] = returns
        buffer_episodes.append(ep_data)

        episode_return = float(rewards.sum())
        mean_risk = float(np.mean(ep_data["risks"]))
        mean_waste = float(np.mean(ep_data["wastes"]))

        metrics = {"policy_loss": None, "value_loss": None, "entropy": None}
        if (episode + 1) % train_cfg["update_every"] == 0:
            rollout_buffer = {
                "obs": torch.tensor(np.concatenate([np.array(e["obs"]) for e in buffer_episodes]), dtype=torch.float32),
                "actions": torch.tensor(np.concatenate([np.array(e["actions"]) for e in buffer_episodes]), dtype=torch.float32),
                "log_probs": torch.tensor(np.concatenate([np.array(e["log_probs"]) for e in buffer_episodes]), dtype=torch.float32),
                "returns": torch.tensor(np.concatenate([e["returns"] for e in buffer_episodes]), dtype=torch.float32),
                "advantages": torch.tensor(np.concatenate([e["advantages"] for e in buffer_episodes]), dtype=torch.float32),
            }
            metrics = agent.update(rollout_buffer)
            buffer_episodes = []

            logger.log_scalar("policy_loss", metrics["policy_loss"], episode)
            logger.log_scalar("value_loss", metrics["value_loss"], episode)
            logger.log_scalar("entropy", metrics["entropy"], episode)

        logger.log_scalar("episode_return", episode_return, episode)
        logger.log_scalar("mean_risk_residual", mean_risk, episode)

        append_csv_row(csv_path, {
            "variant": variant, "seed": seed, "episode": episode, "step": global_step,
            "episode_return": episode_return, "mean_risk_residual": mean_risk,
            "mean_waste": mean_waste, "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"], "entropy": metrics["entropy"],
            "entropy_coef": agent.entropy_coef,
        }, fieldnames)

        if episode_return > best_return:
            best_return = episode_return
            agent.save(os.path.join(log_dir, "best_model.pt"))

        if (episode + 1) % train_cfg["checkpoint_every"] == 0:
            agent.save(os.path.join(log_dir, f"checkpoint_ep{episode+1}.pt"))

        if (episode + 1) % 50 == 0:
            print(f"[{variant} seed {seed}] episode {episode+1}/{train_cfg['n_episodes']} "
                  f"return={episode_return:.2f} risk={mean_risk:.3f}")

    agent.save(os.path.join(log_dir, "final_model.pt"))
    logger.close()
    print(f"[{variant} seed {seed}] terminé. best_return={best_return:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--variant", type=str, default="baseline", choices=["baseline", "enhanced"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["environment"]["seed"]
    train(cfg, seed, args.variant)