"""
reconstruct_train_metrics.py — reconstruit results/train_metrics.csv depuis
les logs TensorBoard (non corrompus).
"""

import os
import csv
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_scalars(log_dir):
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        return None
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    data = {}
    for tag in ["episode_return", "mean_risk_residual", "policy_loss", "value_loss", "entropy"]:
        if tag in ea.Tags().get("scalars", []):
            data[tag] = {e.step: e.value for e in ea.Scalars(tag)}
    return data


def main():
    fieldnames = ["variant", "seed", "episode", "step", "episode_return", "mean_risk_residual",
                  "mean_waste", "policy_loss", "value_loss", "entropy", "entropy_coef"]
    out_path = "results/train_metrics.csv"
    rows = []

    for variant_dir in sorted(glob.glob("logs/*/")):
        variant = os.path.basename(os.path.normpath(variant_dir))
        if variant.startswith("ablation"):
            continue
        for seed_dir in sorted(glob.glob(os.path.join(variant_dir, "seed_*"))):
            seed = int(os.path.basename(seed_dir).replace("seed_", ""))
            data = extract_scalars(seed_dir)
            if not data or "episode_return" not in data:
                print(f"SKIP (pas de donnees): {seed_dir}")
                continue
            episodes = sorted(data["episode_return"].keys())
            for ep in episodes:
                rows.append({
                    "variant": variant, "seed": seed, "episode": ep,
                    "step": (ep + 1) * 30,
                    "episode_return": data["episode_return"].get(ep, ""),
                    "mean_risk_residual": data.get("mean_risk_residual", {}).get(ep, ""),
                    "mean_waste": "",
                    "policy_loss": data.get("policy_loss", {}).get(ep, ""),
                    "value_loss": data.get("value_loss", {}).get(ep, ""),
                    "entropy": data.get("entropy", {}).get(ep, ""),
                    "entropy_coef": "",
                })
            print(f"OK: {variant} seed={seed} -> {len(episodes)} episodes recuperes")

    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{out_path} reconstruit : {len(rows)} lignes au total.")


if __name__ == "__main__":
    main()