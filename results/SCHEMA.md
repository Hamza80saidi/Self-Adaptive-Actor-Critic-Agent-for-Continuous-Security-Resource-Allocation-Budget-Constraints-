# Schéma des fichiers CSV dans results/

Exigence du prof : `results/` doit contenir des `.csv` avec un schéma cohérent.

## `results/train_metrics.csv` (généré par src/train.py)

| Colonne | Type | Description |
|---|---|---|
| `seed` | int | Graine aléatoire du run |
| `episode` | int | Numéro d'épisode |
| `step` | int | Pas de temps global (cumulé sur tous les épisodes) |
| `episode_return` | float | Somme des rewards sur l'épisode |
| `mean_risk_residual` | float | Risque résiduel moyen sur l'épisode |
| `mean_waste` | float | Gaspillage moyen sur l'épisode |
| `policy_loss` | float | Loss de l'Actor à la dernière mise à jour |
| `value_loss` | float | Loss du Critic à la dernière mise à jour |
| `entropy` | float | Entropie de la politique |

## `results/eval_metrics.csv` (généré par src/eval.py)

| Colonne | Type | Description |
|---|---|---|
| `seed` | int | Graine aléatoire du run évalué |
| `agent` | str | `ppo` \| `random` \| `fixed` |
| `episode` | int | Numéro d'épisode d'évaluation |
| `episode_return` | float | Reward cumulé sur l'épisode |
| `mean_risk_residual` | float | Risque résiduel moyen |
| `budget_respected` | bool | Toujours True par construction (test de sanité) |

Ce schéma permet à `src/utils/make_figures.py` de faire un `groupby(step).agg(['mean','std'])`
pour tracer les courbes moyenne +/- écart-type exigées par le prof.
