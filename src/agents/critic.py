"""
critic.py
=========
Réseau de valeur (Critic) pour l'agent actor-critic.

Rôle : estime V(s), la valeur attendue de récompense cumulée à partir de
l'état courant. Utilisé pour calculer l'avantage (GAE) qui guide la mise
à jour de l'Actor.
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    Réseau de valeur V(s).

    Args:
        obs_dim (int): dimension de l'espace d'observation
        hidden_dim (int): taille des couches cachées
        n_hidden_layers (int): nombre de couches cachées
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128, n_hidden_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]  # sortie scalaire = V(s)
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Retourne V(s), la valeur estimée de l'état (scalaire, dim retirée)."""
        return self.net(obs).squeeze(-1)