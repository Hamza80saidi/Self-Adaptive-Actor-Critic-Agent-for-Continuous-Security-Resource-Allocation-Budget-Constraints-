"""TODO Jour 3:
    [ ] Implémenter compute_gae() : calcul de l'avantage généralisé
    [ ] Implémenter update() : boucle d'optimisation PPO clippée
        - ratio = exp(new_log_prob - old_log_prob)
        - surrogate1 = ratio * advantage
        - surrogate2 = clip(ratio, 1-eps, 1+eps) * advantage
        - policy_loss = -min(surrogate1, surrogate2).mean()
        - value_loss = MSE(V(s), returns)
        - entropy_bonus pour favoriser l'exploration
    [ ] Implémenter save()/load() pour les checkpoints
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.actor import Actor
from src.agents.critic import Critic


class PPOAgent:
    """


    Args a faire pour !
        obs_dim (int): dimension de l'observation
        action_dim (int): dimension de l'action (n_resources)
        config (dict): hyperparamètres (voir config/config.yaml -> agent)
    """

    def __init__(self, obs_dim: int, action_dim: int, config: dict):
        self.config = config
        self.actor = Actor(
            obs_dim, action_dim,
            hidden_dim=config["hidden_dim"],
            n_hidden_layers=config["n_hidden_layers"],
        )
        self.critic = Critic(
            obs_dim,
            hidden_dim=config["hidden_dim"],
            n_hidden_layers=config["n_hidden_layers"],
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["actor_lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["critic_lr"])

        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.clip_epsilon = config["clip_epsilon"]
        self.entropy_coef = config["entropy_coef"]
        self.value_loss_coef = config["value_loss_coef"]

    def select_action(self, obs, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # ajoute la dim batch
        with torch.no_grad():  # pas besoin de gradients ici, juste une inférence
            if deterministic:
                action = self.actor.deterministic_action(obs_t)
                log_prob = None
            else:
                action, log_prob = self.actor.sample_action(obs_t)
            value = self.critic(obs_t)

        action_np = action.squeeze(0).numpy()
        log_prob_val = log_prob.item() if log_prob is not None else None
        value_val = value.item()
        return action_np, log_prob_val, value_val

    def compute_gae(self, rewards, values, dones, next_value):
        """
     pour demain :!!

        Args:
            rewards (np.ndarray): shape (T,), reward à chaque pas
            values (np.ndarray): shape (T,), V(s_t) prédit par le Critic à chaque pas
            dones (np.ndarray): shape (T,), 1.0 si l'épisode se termine à ce pas, sinon 0.0
            next_value (float): V(s_T), la valeur du dernier état atteint
                                 (0.0 si l'épisode est vraiment terminé, sinon
                                 l'estimation du Critic pour bootstrap)

        Retourne (advantages, returns), deux np.ndarray de shape (T,).
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        # values_ext = [V(s_0), V(s_1), ..., V(s_{T-1}), V(s_T)]
        values_ext = np.append(values, next_value)

        # On calcule de la FIN vers le DÉBUT (récursion sur last_gae)
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]  # coupe le bootstrap si l'épisode est fini
            delta = rewards[t] + self.gamma * values_ext[t + 1] * mask - values_ext[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values  # returns = ce que le Critic doit apprendre à prédire
        return advantages, returns

    def update(self, rollout_buffer):
        """
        rollout_buffer (dict) doit contenir des torch.Tensor :
            "obs"        : (N, obs_dim)
            "actions"    : (N, action_dim)
            "log_probs"  : (N,)   -- log_prob SOUS L'ANCIENNE politique (au moment de la collecte)
            "returns"    : (N,)
            "advantages" : (N,)

        Retourne un dict de métriques moyennes (utile pour le logging Jour 4).
        """
        obs = rollout_buffer["obs"]
        actions = rollout_buffer["actions"]
        old_log_probs = rollout_buffer["log_probs"]
        returns = rollout_buffer["returns"]
        advantages = rollout_buffer["advantages"]

        # Normalisation des avantages : stabilise énormément l'entraînement
    
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = obs.shape[0]
        batch_size = min(self.config.get("batch_size", 64), n_samples)
        indices = np.arange(n_samples)

        metrics = {"policy_loss": [], "value_loss": [], "entropy": []}

        for _ in range(self.config["n_epochs_per_update"]):
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]

                b_obs = obs[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_returns = returns[batch_idx]
                b_advantages = advantages[batch_idx]

                
                new_log_probs, entropy = self.actor.evaluate_action(b_obs, b_actions)
                ratio = torch.exp(new_log_probs - b_old_log_probs)

                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

               
                new_values = self.critic(b_obs)
                value_loss = nn.functional.mse_loss(new_values, b_returns)

                self.critic_optimizer.zero_grad()
                (self.value_loss_coef * value_loss).backward()
                self.critic_optimizer.step()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.mean().item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def save(self, path: str):
        """Sauvegarde les poids de l'actor et du critic."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        """Charge les poids de l'actor et du critic."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])