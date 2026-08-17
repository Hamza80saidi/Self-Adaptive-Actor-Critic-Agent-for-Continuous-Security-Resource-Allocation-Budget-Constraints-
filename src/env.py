"""
security_env.py
================
Environnement Gymnasium personnalisé pour l'allocation continue de ressources
de sécurité sous contrainte de budget.

CONCEPT GÉNÉRAL
----------------
Chaque jour (pas de temps), l'agent reçoit un budget fixe (total_budget / horizon)
et doit le répartir entre `n_resources` catégories de défense. En parallèle,
`n_threat_types` types de menaces (intrusion réseau, phishing, ransomware...)
ont chacun un niveau d'intensité qui varie de manière stochastique dans le temps.

Chaque ressource a une efficacité différente contre chaque type de menace
(matrice EFFICACY). L'agent doit donc apprendre à REPARTIR son budget de
manière ADAPTATIVE selon les menaces actuellement dominantes -> c'est ce qui
justifie le "Self-Adaptive" du sujet.

Contrainte de budget : l'action brute (logits) est transformée par softmax
en proportions qui somment à 1, puis multipliée par le budget du jour.
=> sum(allocation) == budget_du_jour à CHAQUE step, garanti par construction
   (pas besoin de pénaliser un dépassement, il ne peut pas arriver).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Noms des ressources et types de menaces (pour lisibilité / logs / figures)
RESOURCE_NAMES = [
    "Firewall_IDS",
    "Patch_Management",
    "User_Training",
    "Monitoring_SOC",
    "Backup_Recovery",
]

THREAT_NAMES = [
    "Network_Intrusion",
    "Phishing",
    "Malware_Ransomware",
]

# Matrice d'efficacité : EFFICACY[i, j] = efficacité de la ressource i
# contre la menace j (valeurs entre 0 et 1, choisies selon la littérature
# cybersécurité usuelle -- à justifier/citer dans le rapport si besoin).
#                          Network_Intrusion  Phishing  Ransomware
EFFICACY = np.array([
    [0.90, 0.20, 0.50],   # Firewall_IDS
    [0.60, 0.10, 0.80],   # Patch_Management
    [0.20, 0.90, 0.30],   # User_Training
    [0.70, 0.50, 0.60],   # Monitoring_SOC
    [0.10, 0.10, 0.90],   # Backup_Recovery
])


class SecurityResourceEnv(gym.Env):
    """
    Environnement RL pour l'allocation de ressources de sécurité sous budget.

    Args:
        n_resources (int): nombre de catégories de ressources de sécurité
        horizon (int): nombre de pas de temps (jours) par épisode
        total_budget (float): budget total disponible sur l'épisode
        threat_volatility (float): amplitude de variation stochastique de la menace
        reward_lambda (float): poids de la pénalité de "gaspillage" dans le reward
        seed (int): graine aléatoire pour la reproductibilité
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_resources: int = 5,
        horizon: int = 30,
        total_budget: float = 1000.0,
        threat_volatility: float = 0.3,
        reward_lambda: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        assert n_resources == EFFICACY.shape[0], (
            f"n_resources={n_resources} doit correspondre aux {EFFICACY.shape[0]} "
            f"lignes de la matrice EFFICACY."
        )
        self.n_resources = n_resources
        self.n_threats = EFFICACY.shape[1]
        self.horizon = horizon
        self.total_budget = total_budget
        self.budget_per_day = total_budget / horizon
        self.threat_volatility = threat_volatility
        self.reward_lambda = reward_lambda

        # Action : logits bruts (non normalisés), transformés en proportions
        # via softmax DANS step(). Bornés à [-5, 5] pour la stabilité numérique.
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(n_resources,), dtype=np.float32
        )

        # Observation : [budget_restant_norm, jour_norm, threat_levels(n_threats),
        #                dernière_allocation_norm(n_resources)]
        obs_dim = 2 + self.n_threats + self.n_resources
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._base_seed = seed

        # État interne (initialisé dans reset)
        self.current_day = 0
        self.remaining_budget = None
        self.threat_levels = None
        self.last_allocation = None

    # ------------------------------------------------------------------ #
    # API Gymnasium standard
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        """Réinitialise l'environnement pour un nouvel épisode."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.current_day = 0
        self.remaining_budget = self.total_budget
        # Niveaux de menace initiaux : tirés uniformément dans [0.2, 0.6]
        # (on démarre dans une situation "moyenne", ni calme ni en crise)
        self.threat_levels = self._rng.uniform(0.2, 0.6, size=self.n_threats)
        self.last_allocation = np.zeros(self.n_resources)

        obs = self._get_obs()
        info = {"day": self.current_day}
        return obs, info

    def step(self, action):
        """
        Exécute un pas de temps.

        1. Convertit `action` (logits) en allocation valide via softmax * budget_du_jour
        2. Met à jour le niveau de menace (processus stochastique)
        3. Calcule le risque résiduel et le reward
        4. Avance d'un jour
        5. Vérifie si l'épisode est terminé
        """
        action = np.asarray(action, dtype=np.float64)

        # --- 1. Allocation valide (contrainte budget respectée par construction) ---
        proportions = self._softmax(action)
        allocation = proportions * self.budget_per_day  # somme == budget_per_day

        # --- 2. Mise à jour de la menace AVANT de calculer le risque, pour que
        #        l'agent doive anticiper/réagir à threat_levels observé à t ---
        threat_at_decision = self.threat_levels.copy()
        self._update_threat()

        # --- 3. Risque résiduel et reward ---
        risk_residual = self._compute_risk(allocation, threat_at_decision)
        waste = self._compute_waste(allocation, threat_at_decision)
        reward = -risk_residual - self.reward_lambda * waste

        # --- 4. Avancer d'un jour ---
        self.remaining_budget -= self.budget_per_day
        self.last_allocation = allocation
        self.current_day += 1

        # --- 5. Fin d'épisode ---
        terminated = False  # pas de condition d'échec précoce dans cette version
        truncated = self.current_day >= self.horizon

        obs = self._get_obs()
        info = {
            "day": self.current_day,
            "allocation": allocation,
            "threat_levels": threat_at_decision,
            "risk_residual": risk_residual,
            "waste": waste,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        """Affichage texte simple de l'état courant."""
        print(f"Jour {self.current_day}/{self.horizon} | "
              f"Budget restant: {self.remaining_budget:.1f} | "
              f"Menaces: {dict(zip(THREAT_NAMES, np.round(self.threat_levels, 2)))} | "
              f"Dernière allocation: {dict(zip(RESOURCE_NAMES, np.round(self.last_allocation, 1)))}")

    # ------------------------------------------------------------------ #
    # Méthodes internes
    # ------------------------------------------------------------------ #

    def _get_obs(self):
        """Construit le vecteur d'observation normalisé."""
        # np.clip corrige les erreurs d'arrondi flottant (ex: -1e-16 au lieu
        # de 0.0 sur le dernier pas), qui feraient échouer observation_space.contains()
        budget_norm = np.clip(self.remaining_budget / self.total_budget, 0.0, 1.0)
        day_norm = np.clip(self.current_day / self.horizon, 0.0, 1.0)
        last_alloc_norm = self.last_allocation / self.budget_per_day if self.budget_per_day > 0 else self.last_allocation
        last_alloc_norm = np.clip(last_alloc_norm, 0.0, 1.0)

        obs = np.concatenate([
            [budget_norm],
            [day_norm],
            np.clip(self.threat_levels, 0.0, 1.0),
            last_alloc_norm,
        ]).astype(np.float32)
        return obs

    def _update_threat(self):
        """
        Fait évoluer les niveaux de menace via un processus de type
        Ornstein-Uhlenbeck discrétisé : tendance à revenir vers une moyenne
        (0.4), avec bruit gaussien, plus une petite probabilité de "pic"
        (attaque soudaine) pour rendre l'environnement réaliste et forcer
        l'agent à s'adapter.
        """
        mean_reversion_speed = 0.15
        long_run_mean = 0.4

        noise = self._rng.normal(0, self.threat_volatility, size=self.n_threats)
        drift = mean_reversion_speed * (long_run_mean - self.threat_levels)
        self.threat_levels = self.threat_levels + drift + noise

        # Pic d'attaque aléatoire (5% de chance par jour, par type de menace)
        spike_mask = self._rng.random(self.n_threats) < 0.05
        self.threat_levels = np.where(
            spike_mask,
            np.clip(self.threat_levels + self._rng.uniform(0.3, 0.5, self.n_threats), 0, 1),
            self.threat_levels,
        )

        self.threat_levels = np.clip(self.threat_levels, 0.0, 1.0)

    def _compute_risk(self, allocation, threat_levels):
        """
        Calcule le risque résiduel moyen sur tous les types de menace.

        Pour chaque menace j :
            protection_j = moyenne pondérée de EFFICACY[:, j] par les
                            proportions d'allocation (entre 0 et 1)
            risk_j = threat_levels[j] * (1 - protection_j)

        risk_residual = moyenne des risk_j sur toutes les menaces.
        """
        proportions = allocation / self.budget_per_day  # revient aux proportions [0,1], somme=1
        # protection_par_menace[j] = sum_i EFFICACY[i,j] * proportions[i]
        protection_par_menace = EFFICACY.T @ proportions  # shape (n_threats,)
        risk_par_menace = threat_levels * (1 - protection_par_menace)
        risk_residual = np.mean(risk_par_menace)
        return float(risk_residual)

    def _compute_waste(self, allocation, threat_levels):
        """
        Mesure la fraction du budget dépensée sur des ressources peu utiles
        au vu des menaces ACTUELLES -- pénalise une allocation "statique" qui
        ignore le contexte, et récompense l'adaptivité.

        usefulness[i] = somme pondérée de l'efficacité de la ressource i
                        contre les menaces actuelles (EFFICACY[i,:] . threat_levels)
        waste = proportion du budget allouée à des ressources dont
                l'utilité actuelle est très inférieure à la ressource la
                plus utile en ce moment.
        """
        proportions = allocation / self.budget_per_day
        usefulness = EFFICACY @ threat_levels  # shape (n_resources,)
        max_usefulness = usefulness.max() + 1e-8
        relative_uselessness = 1 - (usefulness / max_usefulness)  # 0 = très utile, 1 = inutile
        waste = float(np.sum(proportions * relative_uselessness))
        return waste

    @staticmethod
    def _softmax(x):
        """Softmax numériquement stable."""
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
