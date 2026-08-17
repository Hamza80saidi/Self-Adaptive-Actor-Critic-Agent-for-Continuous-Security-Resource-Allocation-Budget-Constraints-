"""
run_parallel.py — lance les runs baseline+enhanced en parallèle sur
plusieurs cœurs CPU au lieu de séquentiellement (gain quasi-linéaire avec
le nombre de cœurs, puisque les runs sont indépendants).

Usage: python run_parallel.py --n_workers 6
"""

import argparse
import multiprocessing as mp
import torch

from src.train import load_config, train
from src.eval import evaluate


def worker(args):
    seed, variant, use_annealing = args
    # CRITIQUE : sans ça, chaque process PyTorch essaie d'utiliser TOUS les
    # coeurs pour ses propres calculs matriciels -> contention massive quand
    # plusieurs process tournent en même temps, ça ralentit au lieu d'accélérer
    torch.set_num_threads(1)

    cfg = load_config("configs/config.yaml")
    print(f"[START] {variant} seed={seed}")
    train(cfg, seed, variant=variant, use_annealing=use_annealing)

    checkpoint = f"logs/{variant}/seed_{seed}/best_model.pt"
    evaluate(cfg, checkpoint, seed, variant=variant)
    print(f"[DONE]  {variant} seed={seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=6)
    args = parser.parse_args()

    jobs = []
    for seed in range(5):
        jobs.append((seed, "baseline", False))
        jobs.append((seed, "enhanced", True))

    print(f"Lancement de {len(jobs)} runs sur {args.n_workers} workers en parallèle...")
    with mp.Pool(args.n_workers) as pool:
        pool.map(worker, jobs)

    print("Baseline + Enhanced termines. Lance l'ablation separement (voir message).")