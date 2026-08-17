"""
actor.py
========
Réseau de politique (Actor) pour l'agent actor-critic continu.

Rôle : prend un état (observation) et produit les paramètres d'une
distribution de probabilité continue sur les actions (logits d'allocation,
qui seront transformés en proportions valides par l'environnement via softmax).
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """
    Réseau de politique gaussien pour un espace d'action continu.

    Args:
        obs_dim (int): dimension de l'espace d'observation
        action_dim (int): dimension de l'espace d'action (= n_resources)
        hidden_dim (int): taille des couches cachées
        n_hidden_layers (int): nombre de couches cachées
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, n_hidden_layers: int = 2):
        super().__init__()

        # Backbone : MLP partagé qui extrait des features de l'observation
        layers = [nn.Linear(obs_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.backbone = nn.Sequential(*layers)

        # Tête de sortie : prédit la moyenne (mu) de la gaussienne sur les actions
        self.mu_head = nn.Linear(hidden_dim, action_dim)

        # log_std : paramètre appris, PAS dépendant de l'observation (pratique
        # standard PPO -- un seul écart-type global par dimension d'action,
        # plus stable qu'un log_std prédit par le réseau).
        # Initialisé à 0 -> std = exp(0) = 1.0 au début (exploration large).
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor):
        """Retourne (mu, std) de la distribution gaussienne sur les actions."""
        features = self.backbone(obs)
        mu = self.mu_head(features)
        std = torch.exp(self.log_std)  # exp() garantit std > 0 toujours
        return mu, std

    def sample_action(self, obs: torch.Tensor):
        """
        Échantillonne une action et retourne (action, log_prob) pour
        l'entraînement (exploration stochastique nécessaire pour PPO).
        """
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        action = dist.sample()
        # somme des log_prob sur toutes les dimensions d'action (indépendantes)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor):
        """Retourne l'action moyenne (mu), sans bruit — pour l'évaluation."""
        mu, _ = self.forward(obs)
        return mu

    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Recalcule log_prob et entropie pour une action DÉJÀ prise (utilisé
        pendant la mise à jour PPO, sur des actions collectées avec l'ancienne
        politique -- nécessaire pour calculer le ratio r_t(theta) de PPO).
        """
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy