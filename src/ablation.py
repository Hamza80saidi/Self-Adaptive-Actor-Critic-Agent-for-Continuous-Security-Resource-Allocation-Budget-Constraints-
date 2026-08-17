"""
ablation.py — Section VII du rapport : sensibilité au hyperparametre cle
de l'enhancement (entropy_coef_end). Entraine plusieurs variantes sur les
memes seeds et ecrit results/ablation_metrics.csv.

Usage: python -m src.ablation --config configs/config.yaml
"""

import argparse
import os
import csv
import copy

from src.train import load_config, train
from src.eval import evaluate


def append_csv_row(path, row, fieldnames):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_ablation(config: dict, seeds: list, entropy_ends: dict):
    """
    entropy_ends: dict {nom_variant: valeur_entropy_coef_end}
    ex: {"without_annealing": 0.01 (= pas de decroissance), "full_model": 0.001,
         "aggressive_decay": 0.0001}
    """
    os.makedirs("results", exist_ok=True)
    csv_path = "results/ablation_metrics.csv"
    fieldnames = ["ablation_variant", "entropy_coef_end", "seed", "mean_return", "std_return"]

    for variant_name, end_value in entropy_ends.items():
        cfg = copy.deepcopy(config)
        cfg["enhancement"]["entropy_coef_end"] = end_value

        for seed in seeds:
            log_tag = f"ablation_{variant_name}"
            train(cfg, seed, variant=log_tag, use_annealing=True)
            checkpoint = os.path.join(cfg["training"]["log_dir"], log_tag, f"seed_{seed}", "best_model.pt")
            results = evaluate(cfg, checkpoint, seed, variant=log_tag)

            append_csv_row(csv_path, {
                "ablation_variant": variant_name, "entropy_coef_end": end_value,
                "seed": seed, "mean_return": results[f"ppo_{log_tag}"]["mean_return"],
                "std_return": results[f"ppo_{log_tag}"]["std_return"],
            }, fieldnames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    cfg = load_config(args.config)
    entropy_ends = {
        "without_annealing": cfg["agent"]["entropy_coef"],
        "full_model": cfg["enhancement"]["entropy_coef_end"],
        "aggressive_decay": cfg["enhancement"]["entropy_coef_end"] / 10,
    }
    run_ablation(cfg, args.seeds, entropy_ends)