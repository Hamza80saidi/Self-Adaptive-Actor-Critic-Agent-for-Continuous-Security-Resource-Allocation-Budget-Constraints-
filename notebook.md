# Journal de bord — DS-G05_MD3SI


## Semaine 1 — Environnement, agent, pipeline d'entraînement

### Session 1 — 2026-08-XX

**Claim :** Un environnement avec budget fixe par jour et allocation via softmax
garantirait la contrainte budgétaire sans pénalité de dépassement dans le reward.

**Evidence :** `pytest tests/test_env.py` : 6/6 tests passent, y compris
`test_budget_constraint_respected`. Un bug d'arrondi flottant (`-4.4e-16` au
lieu de `0.0`) détecté par `test_observation_bounds`, corrigé avec `np.clip()`.

**Decision :** Approche softmax conservée (pas de pénalité de dépassement --
inutile, contrainte respectée structurellement). `np.clip()` systématique sur
les observations normalisées.

### Session 2 — 2026-08-XX

**Claim :** L'Actor (gaussien, log_std appris) et le Critic (MLP simple)
suffiraient sans complexité supplémentaire pour un premier prototype PPO.

**Evidence :** Tests manuels cohérents : log_prob recalculé par
`evaluate_action` identique à celui de `sample_action` sur la même action
(vérification critique pour PPO).

**Decision :** Architecture conservée. Pas de partage de poids Actor/Critic.

### Session 3 — 2026-08-XX

**Claim :** GAE + PPO clippé + normalisation des avantages suffiraient pour un
entraînement stable sans tuning supplémentaire.

**Evidence :** Pipeline complet testé (collecte épisode -> GAE -> update PPO)
sans erreur, pertes cohérentes numériquement.

**Decision :** Implémentation conservée. Deux optimizers séparés
(actor_lr=3e-4, critic_lr=1e-3).

### Session 4 — 2026-08-XX

**Claim :** train.py + eval.py + baselines suffiraient à produire des
résultats comparables PPO vs Random vs Fixed.

**Evidence :** Sur 5 seeds réelles (2000 épisodes) : PPO bat clairement les
baselines (mean_return ~-6.2 à -6.4 vs ~-8.1). Bug trouvé et corrigé : les
baselines retournaient `action` seul au lieu de `(action, log_prob, value)`.

**Decision :** Interface unifiée sur les 3 agents. Résultats positifs et
cohérents entre seeds.

---

## Semaine 2 — Alignement sur le guide du prof, enhancement, ablation

### Session 5 — 2026-08-XX

**Claim :** La structure de projet construite initialement correspondait aux
attentes du prof.

**Evidence :** Le guide de rapport (M122) exige un format "Baseline
Reproduction + Enhancement + Ablation" avec IQM et bootstrap CI (rliable),
incompatible avec le plan initial "PPO vs baselines" -- aucune valeur publiée
n'existe pour un environnement synthétique inédit.

**Decision :** Déviation documentée : "Baseline" redéfini comme PPO standard,
"Enhancement" = entropy annealing. `rliable` non installable (conflit de
dépendances) -> réimplémentation manuelle de IQM + bootstrap stratifié dans
`src/utils/stats.py` (méthodologie Agarwal et al. 2021 conservée).

### Session 6 — 2026-08-XX

**Claim :** L'entropy annealing améliorerait le retour final par rapport au
PPO standard.

**Evidence :** Sur 5 seeds, aucune différence mesurable (IQM quasi
identiques, CI95 qui se chevauchent sur les 4 figures). Ablation sur
`entropy_coef_end` (0.0001/0.001/0.01) : résultats indiscernables.

**Decision :** Résultat négatif documenté honnêtement (accepté explicitement
par le guide). Hypothèses proposées pour la Discussion : (1) le baseline
converge déjà vers un plateau stable tôt, (2) le softmax écrase les petites
variations de bruit d'exploration, (3) amplitude testée peut-être trop faible.

### Session 7 — 2026-08-XX

**Claim :** Les vidéos avant/après (barres qui se redessinent chaque jour)
montreraient clairement la différence de comportement.

**Evidence :** Format illisible -- pas de mémoire visuelle d'une frame à
l'autre. Observation additionnelle : l'agent entraîné concentre son budget
sur Firewall_IDS plutôt que de diversifier -- cohérent avec une fonction de
risque LINÉAIRE (pas de rendement décroissant modélisé), dont la solution
optimale mathématique est une solution de coin.

**Decision :** Vidéo reconstruite en aires empilées cumulatives + reward
cumulé superposé (une seule vidéo `comparison.mp4`). Limite de modélisation
(absence de rendement décroissant) documentée pour Discussion / Threats to
validity, non corrigée faute de temps.

---

## Semaine 3 — Rédaction du rapport
### Session 8 — 2026-08-XX

**Claim :** Le format de rapport imposé (guide M122) pourrait être rempli
directement avec les résultats obtenus à 60k steps sans relancer
l'entraînement, en documentant simplement l'écart de budget comme déviation.

**Evidence :** Le prof rappelle explicitement dans les consignes que
"Compute limitations are not an excuse -- plan ahead and use Colab/Kaggle."
Le budget suggéré (~500 000 steps) est donc une exigence à satisfaire
réellement, pas une simple recommandation à documenter comme non atteinte.

**Decision :** Relance complète de l'entraînement (baseline + enhanced +
ablation, 25 runs indépendants) à 500 010 steps réels (16 667 épisodes),
en parallélisant sur plusieurs cœurs CPU pour rester dans un temps
raisonnable (~1h15 au lieu de plusieurs heures en séquentiel). Bug rencontré
et corrigé en cours de route : écritures CSV concurrentes non verrouillées
entre processus parallèles corrompant `results/train_metrics.csv` et
`eval_metrics.csv` -- récupéré sans réentraîner via les logs TensorBoard
individuels (non affectés, un fichier par processus) pour l'entraînement, et
via une réévaluation séquentielle rapide pour l'évaluation.

---

### Session 9 — 2026-08-XX

**Claim :** Les résultats à 60k steps (négatif sur l'entropy annealing)
pourraient changer une fois le budget complet (500k) atteint.

**Evidence :** Résultat négatif confirmé identique à 500k steps : baseline
IQM -5.808 [-5.861,-5.761] vs enhanced IQM -5.853 [-5.985,-5.761], CI
toujours largement chevauchantes. PPO (baseline et enhanced) bat cependant
nettement Random/Fixed (IQM -8.21/-8.23), CI totalement non-chevauchantes --
résultat positif robuste et confirmé à budget complet.

**Decision :** Le résultat négatif sur l'enhancement est traité comme
définitif et informatif (pas un artefact de sous-entraînement, puisqu'il
persiste à budget complet) -- rédigé comme tel dans la section Discussion du
rapport, avec deux mécanismes candidats proposés (plateau précoce de la
courbe d'apprentissage, effet amorti par la saturation du softmax). Analyse
complémentaire de l'allocation par ressource (Fig.5, script
`analyze_allocation.py`) ajoutée pour quantifier -- plutôt que seulement
décrire qualitativement depuis la vidéo -- le comportement de la politique
entraînée : concentration sur 2 ressources (Firewall/IDS, Monitoring/SOC)
plutôt qu'une seule, avec forte variabilité episode à episode.

