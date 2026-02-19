"""Plotting utilities for multi-metric DSBM correlation analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def style_matplotlib() -> None:
    """Apply a publication-oriented plotting style."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _mean_ci(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grouped = df.groupby("gamma")[value_col].agg(["mean", "std", "count"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["ci95"] = 1.96 * grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
    return grouped.sort_values("gamma")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_metric_ami_vs_proxy(
    metric_grid_df: pd.DataFrame,
    overall_corr_row: pd.Series,
    out_path: Path,
) -> None:
    """Scatter plot of AMI vs raw proxy metric values."""
    _ensure_parent(out_path)

    metric_display = str(metric_grid_df["metric_display"].iloc[0])
    objective = str(metric_grid_df["metric_optimize"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    scatter = ax.scatter(
        metric_grid_df["metric_raw"],
        metric_grid_df["ami"],
        c=metric_grid_df["gamma"],
        cmap="viridis",
        s=16,
        alpha=0.42,
        edgecolors="none",
    )

    if len(metric_grid_df) >= 2:
        coeffs = np.polyfit(metric_grid_df["metric_raw"], metric_grid_df["ami"], deg=1)
        x = np.linspace(metric_grid_df["metric_raw"].min(), metric_grid_df["metric_raw"].max(), 220)
        y = coeffs[0] * x + coeffs[1]
        ax.plot(x, y, color="#111111", linewidth=2.0, label="Linear fit")

    text = (
        f"Objective: {'maximize' if objective == 'max' else 'minimize'}\n"
        f"Raw Pearson r={overall_corr_row['pearson_raw']:.3f}\n"
        f"Raw Spearman rho={overall_corr_row['spearman_raw']:.3f}\n"
        f"Aligned Pearson r={overall_corr_row['pearson_aligned']:.3f}\n"
        f"Aligned Spearman rho={overall_corr_row['spearman_aligned']:.3f}\n"
        f"n={int(overall_corr_row['n'])}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03, label="Gamma")
    ax.set_title(f"GSC grid: AMI vs {metric_display}")
    ax.set_xlabel(metric_display)
    ax.set_ylabel("AMI")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_corr_by_gamma(metric_corr_by_gamma_df: pd.DataFrame, out_path: Path) -> None:
    """Correlation-vs-gamma curve for one metric."""
    _ensure_parent(out_path)

    metric_corr_by_gamma_df = metric_corr_by_gamma_df.sort_values("gamma")
    metric_display = str(metric_corr_by_gamma_df["metric_display"].iloc[0])
    objective = str(metric_corr_by_gamma_df["metric_optimize"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    ax.plot(
        metric_corr_by_gamma_df["gamma"],
        metric_corr_by_gamma_df["pearson_aligned"],
        marker="o",
        linewidth=2.0,
        color="#1f77b4",
        label="Pearson (aligned)",
    )
    ax.plot(
        metric_corr_by_gamma_df["gamma"],
        metric_corr_by_gamma_df["spearman_aligned"],
        marker="s",
        linewidth=2.0,
        color="#d62728",
        label="Spearman (aligned)",
    )

    ax.plot(
        metric_corr_by_gamma_df["gamma"],
        metric_corr_by_gamma_df["pearson_raw"],
        marker="o",
        linewidth=1.4,
        linestyle="--",
        color="#1f77b4",
        alpha=0.55,
        label="Pearson (raw)",
    )
    ax.plot(
        metric_corr_by_gamma_df["gamma"],
        metric_corr_by_gamma_df["spearman_raw"],
        marker="s",
        linewidth=1.4,
        linestyle="--",
        color="#d62728",
        alpha=0.55,
        label="Spearman (raw)",
    )

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Correlation with AMI")
    ax.set_title(f"{metric_display}: AMI correlation by gamma ({'maximize' if objective == 'max' else 'minimize'} objective)")
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_gamma_vs_ami(metric_selection_df: pd.DataFrame, out_path: Path) -> None:
    """Plot SC baseline vs GSC selected/oracle AMI by gamma for one metric."""
    _ensure_parent(out_path)

    metric_display = str(metric_selection_df["metric_display"].iloc[0])
    objective = str(metric_selection_df["metric_optimize"].iloc[0])

    fig, ax = plt.subplots(figsize=(9.2, 5.3))

    series = [
        ("sc_ami", "SC-N", "#222222", "--"),
        ("gsc_ami_selected", f"GSC-N selected by {metric_display}", "#1f77b4", "-"),
        ("gsc_ami_oracle", "GSC-N oracle (best AMI)", "#d62728", "-"),
    ]

    for col, label, color, linestyle in series:
        stats = _mean_ci(metric_selection_df, col)
        ax.plot(
            stats["gamma"],
            stats["mean"],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker="o",
            label=label,
        )
        if col != "sc_ami":
            ax.fill_between(
                stats["gamma"],
                stats["mean"] - stats["ci95"],
                stats["mean"] + stats["ci95"],
                color=color,
                alpha=0.14,
                linewidth=0.0,
            )

    ax.set_title(f"AMI vs directionality ({metric_display} proxy, {'max' if objective == 'max' else 'min'} objective)")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("AMI")
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_regret_by_gamma(metric_selection_df: pd.DataFrame, out_path: Path) -> None:
    """Plot selection regret by gamma for one metric."""
    _ensure_parent(out_path)

    metric_display = str(metric_selection_df["metric_display"].iloc[0])
    stats = _mean_ci(metric_selection_df, "selection_regret")

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(stats["gamma"], stats["mean"], marker="o", linewidth=2.0, color="#2ca02c")
    ax.fill_between(
        stats["gamma"],
        stats["mean"] - stats["ci95"],
        stats["mean"] + stats["ci95"],
        color="#2ca02c",
        alpha=0.14,
        linewidth=0.0,
    )
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlabel("Gamma")
    ax.set_ylabel("AMI_oracle - AMI_selected")
    ax.set_title(f"Selection regret by directionality ({metric_display})")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cross_metric_overall_corr(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Compare aligned overall correlations across proxy metrics."""
    _ensure_parent(out_path)
    data = summary_df.copy()

    x = np.arange(len(data))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.bar(x - width / 2, data["overall_pearson_aligned"], width=width, color="#1f77b4", label="Pearson (aligned)")
    ax.bar(x + width / 2, data["overall_spearman_aligned"], width=width, color="#d62728", label="Spearman (aligned)")
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(data["metric_display"], rotation=0)
    ax.set_ylabel("Correlation with AMI")
    ax.set_title("Overall proxy-vs-AMI correlation comparison")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cross_metric_regret(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Compare selection regret across proxy metrics."""
    _ensure_parent(out_path)
    data = summary_df.copy()
    x = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(
        x,
        data["mean_selection_regret"],
        yerr=data["std_selection_regret"],
        capsize=4,
        color="#2ca02c",
        alpha=0.9,
    )
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(data["metric_display"], rotation=0)
    ax.set_ylabel("AMI_oracle - AMI_selected")
    ax.set_title("Proxy model-selection regret comparison")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cross_metric_selected_gain(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Compare GSC(selected)-minus-SC AMI gain across proxy metrics."""
    _ensure_parent(out_path)
    data = summary_df.copy()
    x = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(
        x,
        data["mean_selected_minus_sc"],
        yerr=data["std_selected_minus_sc"],
        capsize=4,
        color="#9467bd",
        alpha=0.9,
    )
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(data["metric_display"], rotation=0)
    ax.set_ylabel("AMI_selected - AMI_SC")
    ax.set_title("GSC(selected)-vs-SC gain comparison")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
