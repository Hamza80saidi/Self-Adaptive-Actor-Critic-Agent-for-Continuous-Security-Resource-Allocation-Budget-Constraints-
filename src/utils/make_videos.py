"""
make_videos.py — UNE seule vidéo comparative, avec des graphes qui
ACCUMULENT l'historique visuellement (pas de barres qui se réinitialisent
à chaque frame) :
  - Panel haut  : allocation empilée (stacked area) - Avant (agent random)
  - Panel milieu: allocation empilée (stacked area) - Apres (agent PPO)
  - Panel bas   : reward cumule des deux, superpose -> l'ecart se voit
                  se creuser progressivement

Usage: python -m src.utils.make_videos --config configs/config.yaml --checkpoint logs/enhanced/seed_0/best_model.pt --out_dir videos/
"""

import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.env import SecurityResourceEnv, RESOURCE_NAMES, THREAT_NAMES
from src.agents.actor_critic import PPOAgent
from src.baselines.random_agent import RandomAgent

# Utilise le binaire ffmpeg embarque par imageio-ffmpeg si aucun ffmpeg
# systeme n'est trouve dans le PATH (evite d'avoir a l'installer/PATH sous Windows).
import shutil
if shutil.which("ffmpeg") is None:
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("ATTENTION: ffmpeg introuvable et imageio-ffmpeg non installe.")
        print("Lance: python -m pip install imageio-ffmpeg")


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_episode_and_record(env, agent, seed, deterministic=True):
    """Déroule un épisode avec une seed FIXE (menaces identiques d'un run à l'autre)."""
    obs, _ = env.reset(seed=seed)
    allocations, threats, cum_rewards = [], [], []
    total = 0.0
    for _ in range(env.horizon):
        action, _, _ = agent.select_action(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        allocations.append(info["allocation"])
        threats.append(info["threat_levels"])
        total += reward
        cum_rewards.append(total)
    return {
        "allocations": np.array(allocations),
        "threats": np.array(threats),
        "cum_rewards": np.array(cum_rewards),
    }


def make_comparison_video(hist_before, hist_after, out_path, max_budget):
    n_days = hist_before["allocations"].shape[0]
    days = np.arange(1, n_days + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(RESOURCE_NAMES)))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    def update(t):
        ax1.clear(); ax2.clear(); ax3.clear()
        x = days[:t + 1]

        ax1.stackplot(x, *hist_before["allocations"][:t + 1].T, labels=RESOURCE_NAMES, colors=colors)
        ax1.set_ylim(0, max_budget)
        ax1.set_xlim(1, n_days)
        ax1.set_ylabel("Allocation")
        ax1.set_title("AVANT entrainement (agent aleatoire)", loc="left", fontsize=10)

        ax2.stackplot(x, *hist_after["allocations"][:t + 1].T, labels=RESOURCE_NAMES, colors=colors)
        ax2.set_ylim(0, max_budget)
        ax2.set_xlim(1, n_days)
        ax2.set_ylabel("Allocation")
        ax2.set_title("APRES entrainement (PPO)", loc="left", fontsize=10)
        ax2.legend(loc="upper left", fontsize=6, ncol=5, bbox_to_anchor=(0, -0.15))

        ax3.plot(x, hist_before["cum_rewards"][:t + 1], color="gray", linewidth=2, label="Avant")
        ax3.plot(x, hist_after["cum_rewards"][:t + 1], color="steelblue", linewidth=2, label="Apres")
        ax3.set_xlim(1, n_days)
        ax3.set_ylim(min(hist_before["cum_rewards"].min(), hist_after["cum_rewards"].min()) * 1.1, 0)
        ax3.set_xlabel("Jour")
        ax3.set_ylabel("Reward cumule")
        ax3.legend(loc="lower left", fontsize=8)
        ax3.grid(alpha=0.3)

        fig.suptitle(f"Jour {t + 1}/{n_days}  |  ecart de reward cumule = "
                     f"{hist_after['cum_rewards'][t] - hist_before['cum_rewards'][t]:+.2f}",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

    anim = animation.FuncAnimation(fig, update, frames=n_days)
    anim.save(out_path, writer="ffmpeg", fps=2)
    plt.close(fig)
    print(f"Video sauvegardee: {out_path}")


def main(config_path, checkpoint, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config(config_path)
    env_cfg = cfg["environment"]
    agent_cfg = cfg["agent"]
    video_seed = env_cfg["seed"]

    env = SecurityResourceEnv(
        n_resources=env_cfg["n_resources"], horizon=env_cfg["horizon"],
        total_budget=env_cfg["total_budget"], threat_volatility=env_cfg["threat_volatility"],
        reward_lambda=env_cfg["reward_lambda"], seed=video_seed,
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    random_agent = RandomAgent(env.n_resources, seed=video_seed)
    hist_before = run_episode_and_record(env, random_agent, seed=video_seed, deterministic=False)

    ppo_agent = PPOAgent(obs_dim, action_dim, agent_cfg)
    ppo_agent.load(checkpoint)
    hist_after = run_episode_and_record(env, ppo_agent, seed=video_seed, deterministic=True)

    max_budget = env.budget_per_day * 1.05
    make_comparison_video(hist_before, hist_after, os.path.join(out_dir, "comparison.mp4"), max_budget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="logs/enhanced/seed_0/best_model.pt")
    parser.add_argument("--out_dir", type=str, default="videos/")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.out_dir)