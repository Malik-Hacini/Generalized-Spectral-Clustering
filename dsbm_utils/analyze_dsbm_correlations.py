"""
Analyze DSBM benchmark correlations and generate publication-quality plots.

Produces two analyses from `results/benchmark_dsbm_grid_search`:
1) Correlation between gamma and AMI for SC-N vs GSC-N.
2) Correlation between AMI and Graph-CH across the GSC-N grid search.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


DATASET_RE = re.compile(r"^dsbm_gamma(?P<gamma>\d+(?:\.\d+)?)_seed(?P<seed>\d+)$")


def _format_p_value(p_value: float) -> str:
    if p_value < 1e-4:
        return "<1e-4"
    return f"{p_value:.4f}"


def _style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _iter_gamma_dataset_dirs(results_dir: Path):
    for path in sorted(results_dir.iterdir()):
        if not path.is_dir():
            continue
        match = DATASET_RE.match(path.name)
        if match is None:
            continue
        gamma = float(match.group("gamma"))
        seed = int(match.group("seed"))
        yield path, gamma, seed


def load_summary_records(results_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for dataset_dir, gamma, seed in _iter_gamma_dataset_dirs(results_dir):
        summary_path = dataset_dir / f"{dataset_dir.name}_summary.csv"
        if not summary_path.exists():
            continue

        summary_df = pd.read_csv(summary_path)
        for method in ("SC-N", "GSC-N"):
            method_rows = summary_df[summary_df["method"] == method]
            if method_rows.empty:
                continue
            row = method_rows.iloc[0]
            records.append(
                {
                    "dataset": dataset_dir.name,
                    "gamma": gamma,
                    "seed": seed,
                    "method": method,
                    "ami": float(row["ami_best_mean"]),
                    "graph_ch": float(row["graph_ch_best_mean"]),
                }
            )

    if not records:
        raise RuntimeError("No gamma summary records found. Run benchmark_dsbm.py first.")
    return pd.DataFrame(records)


def load_gsc_grid_records(results_dir: Path, method: str = "GSC-N") -> pd.DataFrame:
    records: list[dict] = []
    for dataset_dir, gamma, seed in _iter_gamma_dataset_dirs(results_dir):
        all_results_path = dataset_dir / method / f"{method}_all_results.json"
        if not all_results_path.exists():
            continue

        with all_results_path.open("r", encoding="utf-8") as f:
            all_results = json.load(f)

        for entry in all_results:
            measure = entry.get("measure", [None, {}])
            measure_kwargs = {}
            if isinstance(measure, list) and len(measure) >= 2 and isinstance(measure[1], dict):
                measure_kwargs = measure[1]

            records.append(
                {
                    "dataset": dataset_dir.name,
                    "gamma": gamma,
                    "seed": seed,
                    "alpha": float(measure_kwargs.get("alpha", np.nan)),
                    "t": float(measure_kwargs.get("t", np.nan)),
                    "ami": float(entry["ami"]["mean"]),
                    "graph_ch": float(entry["graph_ch"]["mean"]),
                }
            )

    if not records:
        raise RuntimeError("No GSC-N grid records found. Run benchmark_dsbm.py first.")
    return pd.DataFrame(records)


def compute_gamma_ami_correlation(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for method in ("SC-N", "GSC-N"):
        data = summary_df[summary_df["method"] == method]
        pearson_r, pearson_p = pearsonr(data["gamma"], data["ami"])
        spearman_rho, spearman_p = spearmanr(data["gamma"], data["ami"])
        out.append(
            {
                "analysis": "gamma_vs_ami",
                "method": method,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": float(spearman_rho),
                "spearman_p": float(spearman_p),
                "n": len(data),
            }
        )
    return pd.DataFrame(out)


def compute_ami_graph_ch_correlation(gsc_grid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson_r, pearson_p = pearsonr(gsc_grid_df["graph_ch"], gsc_grid_df["ami"])
    spearman_rho, spearman_p = spearmanr(gsc_grid_df["graph_ch"], gsc_grid_df["ami"])

    overall = pd.DataFrame(
        [
            {
                "analysis": "ami_vs_graph_ch_gsc_grid",
                "scope": "overall",
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": float(spearman_rho),
                "spearman_p": float(spearman_p),
                "n": len(gsc_grid_df),
            }
        ]
    )

    by_gamma = []
    for gamma, gamma_df in sorted(gsc_grid_df.groupby("gamma"), key=lambda x: x[0]):
        pr, pp = pearsonr(gamma_df["graph_ch"], gamma_df["ami"])
        sr, sp = spearmanr(gamma_df["graph_ch"], gamma_df["ami"])
        by_gamma.append(
            {
                "gamma": gamma,
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_rho": float(sr),
                "spearman_p": float(sp),
                "n": len(gamma_df),
            }
        )

    return overall, pd.DataFrame(by_gamma)


def plot_gamma_vs_ami(summary_df: pd.DataFrame, stats_df: pd.DataFrame, out_path: Path) -> None:
    colors = {"SC-N": "#1f77b4", "GSC-N": "#d62728"}
    labels = {"SC-N": "Classical SC (SC-N)", "GSC-N": "Generalized SC (GSC-N)"}

    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    rng = np.random.default_rng(42)

    for method in ("SC-N", "GSC-N"):
        method_df = summary_df[summary_df["method"] == method].copy()
        jitter = rng.uniform(-0.02, 0.02, size=len(method_df))
        ax.scatter(
            method_df["gamma"] + jitter,
            method_df["ami"],
            s=38,
            alpha=0.40,
            color=colors[method],
            edgecolors="none",
        )

        grouped = (
            method_df.groupby("gamma")["ami"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("gamma")
        )
        se = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"])  # type: ignore[arg-type]
        ci95 = 1.96 * se

        ax.plot(
            grouped["gamma"],
            grouped["mean"],
            color=colors[method],
            linewidth=2.5,
            marker="o",
            markersize=6,
            label=labels[method],
        )
        ax.fill_between(
            grouped["gamma"],
            grouped["mean"] - ci95,
            grouped["mean"] + ci95,
            color=colors[method],
            alpha=0.15,
            linewidth=0,
        )

    ax.set_title("DSBM Directionality vs Clustering Quality")
    ax.set_xlabel("Directionality parameter $\\gamma$")
    ax.set_ylabel("AMI (best over method grid)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.legend(frameon=False, loc="upper right")

    stats_lines = []
    for method in ("SC-N", "GSC-N"):
        row = stats_df[stats_df["method"] == method].iloc[0]
        stats_lines.append(
            f"{method}: r={row['pearson_r']:.3f} (p={_format_p_value(row['pearson_p'])}), "
            f"rho={row['spearman_rho']:.3f}"
        )

    ax.text(
        0.02,
        0.02,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_ami_vs_graph_ch(gsc_grid_df: pd.DataFrame, overall_df: pd.DataFrame, by_gamma_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax_scatter, ax_corr = axes

    scatter = ax_scatter.scatter(
        gsc_grid_df["graph_ch"],
        gsc_grid_df["ami"],
        c=gsc_grid_df["gamma"],
        cmap="viridis",
        s=16,
        alpha=0.42,
        edgecolors="none",
    )

    coef = np.polyfit(gsc_grid_df["graph_ch"], gsc_grid_df["ami"], deg=1)
    x_line = np.linspace(gsc_grid_df["graph_ch"].min(), gsc_grid_df["graph_ch"].max(), 250)
    y_line = coef[0] * x_line + coef[1]
    ax_scatter.plot(x_line, y_line, color="black", linewidth=2.2, label="Linear fit")

    row = overall_df.iloc[0]
    text = (
        f"Overall: r={row['pearson_r']:.3f} (p={_format_p_value(row['pearson_p'])})\n"
        f"rho={row['spearman_rho']:.3f} (p={_format_p_value(row['spearman_p'])})\n"
        f"n={int(row['n'])} grid points"
    )
    ax_scatter.text(
        0.02,
        0.98,
        text,
        transform=ax_scatter.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    cbar = fig.colorbar(scatter, ax=ax_scatter, fraction=0.046, pad=0.03)
    cbar.set_label("Dataset directionality $\\gamma$")

    ax_scatter.set_title("GSC Grid Search: AMI vs Graph-CH")
    ax_scatter.set_xlabel("Graph-CH")
    ax_scatter.set_ylabel("AMI")
    ax_scatter.legend(frameon=False, loc="lower right")

    by_gamma_df = by_gamma_df.sort_values("gamma")
    ax_corr.plot(
        by_gamma_df["gamma"],
        by_gamma_df["pearson_r"],
        marker="o",
        linewidth=2.0,
        color="#2ca02c",
        label="Pearson r",
    )
    ax_corr.plot(
        by_gamma_df["gamma"],
        by_gamma_df["spearman_rho"],
        marker="s",
        linewidth=2.0,
        color="#ff7f0e",
        label="Spearman rho",
    )
    ax_corr.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.set_xticks(np.linspace(0.0, 1.0, 6))
    ax_corr.set_xlabel("$\\gamma$")
    ax_corr.set_ylabel("Correlation coefficient")
    ax_corr.set_title("Correlation Strength by Directionality")
    ax_corr.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style_matplotlib()

    results_dir = Path("results/benchmark_dsbm_grid_search")
    out_dir = results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary_records(results_dir)
    gsc_grid_df = load_gsc_grid_records(results_dir, method="GSC-N")

    gamma_ami_stats = compute_gamma_ami_correlation(summary_df)
    ami_graph_overall, ami_graph_by_gamma = compute_ami_graph_ch_correlation(gsc_grid_df)

    plot_gamma_vs_ami(summary_df, gamma_ami_stats, out_dir / "gamma_vs_ami_sc_vs_gsc")
    plot_ami_vs_graph_ch(
        gsc_grid_df,
        ami_graph_overall,
        ami_graph_by_gamma,
        out_dir / "gsc_grid_ami_vs_graph_ch",
    )

    gamma_ami_stats.to_csv(out_dir / "gamma_vs_ami_correlation.csv", index=False)
    ami_graph_overall.to_csv(out_dir / "gsc_grid_ami_vs_graph_ch_overall.csv", index=False)
    ami_graph_by_gamma.to_csv(out_dir / "gsc_grid_ami_vs_graph_ch_by_gamma.csv", index=False)

    print("Saved analysis artifacts to:", out_dir)
    print("  - gamma_vs_ami_sc_vs_gsc.(png|pdf)")
    print("  - gsc_grid_ami_vs_graph_ch.(png|pdf)")
    print("  - gamma_vs_ami_correlation.csv")
    print("  - gsc_grid_ami_vs_graph_ch_overall.csv")
    print("  - gsc_grid_ami_vs_graph_ch_by_gamma.csv")


if __name__ == "__main__":
    main()
