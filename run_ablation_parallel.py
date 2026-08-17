"""
run_ablation_parallel.py — lance les 15 runs de l'ablation (3 variantes x 5
seeds) en parallele sur plusieurs coeurs CPU.

Usage: python run_ablation_parallel.py --n_workers 6
"""

import argparse
import os
import csv
import multiprocessing as mp
import torch

from src.train import load_config, train
from src.eval import evaluate


def worker(args):
    seed, variant_name, entropy_end = args
    torch.set_num_threads(1)  # evite la contention entre process paralleles

    cfg = load_config("configs/config.yaml")
    cfg["enhancement"]["entropy_coef_end"] = entropy_end

    log_tag = f"ablation_{variant_name}"
    print(f"[START] {log_tag} seed={seed} (entropy_coef_end={entropy_end})")
    train(cfg, seed, variant=log_tag, use_annealing=True)

    checkpoint = f"logs/{log_tag}/seed_{seed}/best_model.pt"
    results = evaluate(cfg, checkpoint, seed, variant=log_tag)
    print(f"[DONE]  {log_tag} seed={seed}")
    return (variant_name, entropy_end, seed, results[f"ppo_{log_tag}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config("configs/config.yaml")
    entropy_ends = {
        "without_annealing": cfg["agent"]["entropy_coef"],
        "full_model": cfg["enhancement"]["entropy_coef_end"],
        "aggressive_decay": cfg["enhancement"]["entropy_coef_end"] / 10,
    }

    jobs = [(seed, name, end) for name, end in entropy_ends.items() for seed in range(5)]

    print(f"Lancement de {len(jobs)} runs d'ablation sur {args.n_workers} workers...")
    with mp.Pool(args.n_workers) as pool:
        results = pool.map(worker, jobs)

    os.makedirs("results", exist_ok=True)
    csv_path = "results/ablation_metrics.csv"
    fieldnames = ["ablation_variant", "entropy_coef_end", "seed", "mean_return", "std_return"]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for variant_name, end_value, seed, res in results:
            writer.writerow({
                "ablation_variant": variant_name, "entropy_coef_end": end_value,
                "seed": seed, "mean_return": res["mean_return"], "std_return": res["std_return"],
            })

    print("Ablation terminee. results/ablation_metrics.csv ecrit.")