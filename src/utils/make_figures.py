"""
make_figures.py — 4 figures exigees par le guide du prof (Section VI-VII) :
  Fig 1 : courbes d'apprentissage, 5 seeds, baseline + enhanced, CI95 ombree
  Fig 2 : comparaison IQM baseline vs enhanced, bootstrap CI95
  Fig 3 : performance profile (fraction de runs >= seuil, par variant)
  Fig 4 : sensibilite au hyperparametre cle de l'enhancement (ablation)

Usage: python -m src.utils.make_figures --results_dir results/ --out_dir figures/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.stats import iqm, stratified_bootstrap_ci


def fig1_training_curves(df_train, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, color in [("baseline", "gray"), ("enhanced", "steelblue")]:
        sub = df_train[df_train["variant"] == variant]
        if sub.empty:
            continue
        agg = sub.groupby("episode")["episode_return"].agg(["mean", "std", "count"]).reset_index()
        ci95 = 1.96 * agg["std"] / np.sqrt(agg["count"])
        ax.plot(agg["episode"], agg["mean"], label=variant, color=color)
        ax.fill_between(agg["episode"], agg["mean"] - ci95, agg["mean"] + ci95, alpha=0.25, color=color)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.set_title("Fig. 1 - Courbes d'apprentissage (5 seeds, CI95 ombree)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def fig2_iqm_comparison(df_eval, out_path):
    variants = []
    points, lowers, uppers = [], [], []
    for variant in ["baseline", "enhanced"]:
        agent_name = f"ppo_{variant}"
        sub = df_eval[df_eval["agent"] == agent_name]
        if sub.empty:
            continue
        returns_by_seed = [g["episode_return"].values for _, g in sub.groupby("seed")]
        point, lower, upper = stratified_bootstrap_ci(returns_by_seed)
        variants.append(variant)
        points.append(point); lowers.append(point - lower); uppers.append(upper - point)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(variants, points, yerr=[lowers, uppers], capsize=8, color=["gray", "steelblue"][:len(variants)])
    ax.set_ylabel("IQM (episode return)")
    ax.set_title("Fig. 2 - IQM baseline vs enhanced (bootstrap CI95)")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def fig3_performance_profile(df_eval, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    all_returns = df_eval[df_eval["agent"].isin(["ppo_baseline", "ppo_enhanced"])]["episode_return"]
    if all_returns.empty:
        plt.close(fig)
        return
    taus = np.linspace(all_returns.min(), all_returns.max(), 100)

    for variant, color in [("baseline", "gray"), ("enhanced", "steelblue")]:
        sub = df_eval[df_eval["agent"] == f"ppo_{variant}"]["episode_return"].values
        if len(sub) == 0:
            continue
        fractions = [np.mean(sub >= t) for t in taus]
        ax.plot(taus, fractions, label=variant, color=color)

    ax.set_xlabel("Seuil de retour (tau)")
    ax.set_ylabel("Fraction des runs >= tau")
    ax.set_title("Fig. 3 - Performance profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def fig4_sensitivity(df_ablation, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    agg = df_ablation.groupby(["ablation_variant", "entropy_coef_end"])["mean_return"].agg(["mean", "std"]).reset_index()
    agg = agg.sort_values("entropy_coef_end")
    ax.errorbar(agg["entropy_coef_end"].astype(str), agg["mean"], yerr=agg["std"], fmt="o-", capsize=5, color="steelblue")
    ax.set_xlabel("entropy_coef_end (hyperparametre cle)")
    ax.set_ylabel("Mean return (across seeds)")
    ax.set_title("Fig. 4 - Sensibilite au hyperparametre de l'enhancement")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def main(results_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_csv = os.path.join(results_dir, "train_metrics.csv")
    eval_csv = os.path.join(results_dir, "eval_metrics.csv")
    ablation_csv = os.path.join(results_dir, "ablation_metrics.csv")

    if os.path.isfile(train_csv):
        fig1_training_curves(pd.read_csv(train_csv), os.path.join(out_dir, "fig1_training_curves.pdf"))
    else:
        print(f"Absent: {train_csv}")

    if os.path.isfile(eval_csv):
        df_eval = pd.read_csv(eval_csv)
        fig2_iqm_comparison(df_eval, os.path.join(out_dir, "fig2_iqm_comparison.pdf"))
        fig3_performance_profile(df_eval, os.path.join(out_dir, "fig3_performance_profile.pdf"))
    else:
        print(f"Absent: {eval_csv}")

    if os.path.isfile(ablation_csv):
        fig4_sensitivity(pd.read_csv(ablation_csv), os.path.join(out_dir, "fig4_sensitivity.pdf"))
    else:
        print(f"Absent: {ablation_csv} -- lance src/ablation.py d'abord")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--out_dir", type=str, default="figures/")
    args = parser.parse_args()
    main(args.results_dir, args.out_dir)