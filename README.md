# DS-G05_MD3SI — Self-Adaptive Actor-Critic Agent for Continuous Security Resource Allocation under Budget Constraints

Projet de fin de module — RL — Master M1 Data Science & Sécurité des Systèmes d'Information
Planning réel : 3 semaines.
> ⚠️ **Prérequis obligatoire : Python 3.10, 3.11 ou 3.12.**
> Python 3.13+ fait échouer `pip install -r requirements.txt` (scipy n'a pas
> encore de wheels précompilées pour ces versions sur PyPI ; pip tente alors
> de compiler depuis les sources, ce qui échoue sans compilateur Fortran
> installé — erreur typique : `Unknown compiler(s): [ifort, gfortran, ...]`).
> Vérifie ta version avant de commencer :
> ```bash
> python --version
> ```
> Si elle affiche 3.13 ou plus, installe Python 3.11 séparément
> (https://www.python.org/downloads/release/python-3110/) et crée le venv
> avec cette version précise :
> ```bash
> py -3.11 -m venv venv        # Windows
> python3.11 -m venv venv      # Linux/Mac
> ```

## Statut de ce dépôt

Ce dépôt est poussé **avec les résultats finaux inclus** (`logs/`, `results/`,
`figures/`, `videos/`) --- budget complet (~500 000 steps/run), 5 seeds,
25 runs (baseline + enhanced + ablation). Tu peux consulter directement ces
fichiers sans rien relancer.

**Si tu veux relancer le projet depuis zéro** (recalculer plutôt que
consulter), supprime d'abord les résultats existants pour éviter que les
nouveaux runs ne se mélangent avec les anciens dans les mêmes fichiers CSV :

```bash
rm -rf logs/* results/* figures/* videos/*
```

Puis suis la section ## "Reproduire tous les résultats" en BAS  .
## Reproduire tous les résultats

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` régénère : entraînement (baseline + enhanced, 5 seeds chacun),
évaluation, ablation, figures (`figures/*.pdf`), vidéo (`videos/comparison.mp4`).


## Entraînement parallèle (optionnel, plus rapide en local)

`run_all.sh` exécute tout séquentiellement (~5h, portable partout : Colab,
n'importe quelle machine). En local sur une machine multi-cœurs, deux
scripts parallélisent les runs indépendants et réduisent le temps à
~1h15 :

```bash
python run_parallel.py --n_workers 6           # baseline + enhanced (10 runs)
python run_ablation_parallel.py --n_workers 8   # ablation (15 runs)
```

Ajuste `--n_workers` au nombre de cœurs de ta machine (laisse 1-2 cœurs
libres pour l'OS).

**Attention** : ces deux scripts écrivent chacun leurs runs en parallèle
dans `logs/`, mais `results/train_metrics.csv` est écrit de façon non
sûre en concurrence (écritures simultanées non verrouillées, peut corrompre
le fichier). Si `results/train_metrics.csv` devient illisible
(`pandas.errors.ParserError`), reconstruis-le depuis les logs TensorBoard
individuels (non affectés, un fichier par run) au lieu de réentraîner :

```bash
python reconstruct_train_metrics.py
```

Si `results/eval_metrics.csv` est corrompu (même cause), il n'y a pas
d'équivalent TensorBoard à récupérer -- relance juste l'évaluation
(rapide, pas de réentraînement, checkpoints déjà sauvegardés), en
**séquentiel** cette fois pour éviter de recréer le bug :

```bash
rm -f results/eval_metrics.csv
for seed in 0 1 2 3 4; do
    python -m src.eval --config configs/config.yaml --checkpoint logs/baseline/seed_${seed}/best_model.pt --seed $seed --variant baseline
    python -m src.eval --config configs/config.yaml --checkpoint logs/enhanced/seed_${seed}/best_model.pt --seed $seed --variant enhanced
done
```
## Résumé du problème

Un agent RL alloue quotidiennement un budget de sécurité limité entre 5
catégories de ressources défensives (firewall/IDS, patch management,
formation utilisateurs, monitoring/SOC, backup/recovery), face à 3 types de
menace dont l'intensité varie de manière stochastique. Objectif : minimiser
le risque résiduel cumulé sur un horizon de 30 jours, sous contrainte stricte
de budget.

**Important :** `SecurityResourceEnv` est un environnement **synthétique**
inventé pour ce projet -- pas de valeur publiée existante à reproduire (voir
"Déviations" ci-dessous).

## Choix de conception

| Aspect | Choix | Justification |
|---|---|---|
| Horizon | 30 jours | Assez long pour observer l'adaptation aux pics de menace |
| Espace d'action | Continu (5 logits) -> softmax x budget du jour | Contrainte budgétaire respectée par construction |
| Reward | `-risque_résiduel - lambda*gaspillage` | Composite, permet une ablation |
| Algorithme | PPO continu (actor-critic) | Stable, robuste, adapté au format du module |
| Baseline | PPO standard (entropy_coef fixe) | Pas de valeur publiée disponible pour cet environnement inédit |
| Enhancement | Entropy annealing (0.01 -> 0.001) | Modification isolée, ablation sur `entropy_coef_end` |

## Structure du projet

```text
DS-G05_MD3SI/
├── README.md
├── requirements.txt
├── run_all.sh                       # pipeline séquentiel officiel (reproduction)
├── run_parallel.py                  # baseline+enhanced en parallèle (confort local)
├── run_ablation_parallel.py         # ablation en parallèle (confort local)
├── reconstruct_train_metrics.py     # récupère train_metrics.csv depuis TensorBoard si corrompu
├── report.pdf                        # rapport final (racine, exigé par le prof)
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── env.py
│   ├── train.py                     # --variant baseline|enhanced
│   ├── eval.py                      # --variant baseline|enhanced
│   ├── ablation.py                  # sweep entropy_coef_end
│   ├── agents/
│   │   ├── actor.py
│   │   ├── critic.py
│   │   └── actor_critic.py
│   ├── baselines/
│   │   ├── random_agent.py
│   │   └── fixed_agent.py
│   └── utils/
│       ├── seed.py
│       ├── logger.py
│       ├── stats.py                 # IQM + bootstrap stratifié (remplace rliable)
│       ├── make_figures.py          # fig1 à fig4
│       ├── make_videos.py           # comparison.mp4
│       └── analyze_allocation.py    # fig5
│
├── results/
│   ├── SCHEMA.md
│   ├── train_metrics.csv
│   ├── eval_metrics.csv
│   └── ablation_metrics.csv
│
├── logs/
│   ├── baseline/seed_0.../seed_4/
│   ├── enhanced/seed_0.../seed_4/
│   └── ablation_without_annealing/, ablation_full_model/, ablation_aggressive_decay/ (seed_0..4 chacun)
│
├── figures/
│   ├── fig1_training_curves.pdf
│   ├── fig2_iqm_comparison.pdf
│   ├── fig3_performance_profile.pdf
│   ├── fig4_sensitivity.pdf
│   └── fig5_allocation_before_after.pdf
│
├── videos/
│   └── comparison.mp4
│
├── tests/
│   └── test_env.py
│
├── report/
│   └── report.tex                   # source LaTeX (le .pdf compilé est à la racine)
│
└── notebook.md
