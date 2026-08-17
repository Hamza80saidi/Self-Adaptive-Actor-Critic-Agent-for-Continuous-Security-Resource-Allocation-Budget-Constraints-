#!/bin/bash
# run_all.sh — régénère TOUT le projet en une seule commande.
set -e

CONFIG="configs/config.yaml"
SEEDS=(0 1 2 3 4)

echo "=== [1/5] Entrainement baseline + enhanced, multi-seed ==="
for seed in "${SEEDS[@]}"; do
    echo "--- seed $seed ---"
    python -m src.train --config "$CONFIG" --seed "$seed" --variant baseline
    python -m src.train --config "$CONFIG" --seed "$seed" --variant enhanced
done

echo "=== [2/5] Evaluation (PPO baseline/enhanced vs Random/Fixed) ==="
for seed in "${SEEDS[@]}"; do
    python -m src.eval --config "$CONFIG" --checkpoint "logs/baseline/seed_${seed}/best_model.pt" --seed "$seed" --variant baseline
    python -m src.eval --config "$CONFIG" --checkpoint "logs/enhanced/seed_${seed}/best_model.pt" --seed "$seed" --variant enhanced
done

echo "=== [3/5] Ablation (sensibilite entropy_coef_end) ==="
python -m src.ablation --config "$CONFIG" --seeds "${SEEDS[@]}"

echo "=== [4/5] Figures (4 PDF, IQM + bootstrap CI95) ==="
python -m src.utils.make_figures --results_dir results/ --out_dir figures/

echo "=== [5/5] Videos avant/apres ==="
python -m src.utils.make_videos --config "$CONFIG" --checkpoint "logs/enhanced/seed_0/best_model.pt" --out_dir videos/

echo "=== Termine. Voir results/, figures/, videos/, logs/ ==="