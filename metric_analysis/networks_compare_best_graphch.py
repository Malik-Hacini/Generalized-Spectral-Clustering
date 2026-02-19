"""Compare best Graph-CH profile against modularity/map-equation on networks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _style() -> None:
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


def build_comparison_df(graphch_analysis_dir: Path, other_analysis_dir: Path) -> pd.DataFrame:
    best_path = graphch_analysis_dir / "manifests" / "best_profile.json"
    profile_summary_path = graphch_analysis_dir / "tables" / "profile_summary.csv"
    other_summary_path = other_analysis_dir / "tables" / "summary_by_metric.csv"

    if not best_path.exists() or not profile_summary_path.exists() or not other_summary_path.exists():
        raise FileNotFoundError("Missing one of required analysis files for comparison")

    best_payload = json.loads(best_path.read_text())
    best_profile_id = str(best_payload["best_profile"]["profile_id"])

    profile_summary_df = pd.read_csv(profile_summary_path)
    row_best = profile_summary_df[profile_summary_df["profile_id"] == best_profile_id]
    if row_best.empty:
        raise RuntimeError(f"Best profile '{best_profile_id}' not found in profile_summary.csv")
    row_best = row_best.iloc[0]

    rows = [
        {
            "metric": "graph_ch",
            "metric_display": f"Graph-CH ({best_profile_id})",
            "metric_optimize": "max",
            "overall_spearman_aligned": float(row_best["overall_spearman"]),
            "overall_pearson_aligned": float(row_best["overall_pearson"]),
            "mean_selection_regret": float(row_best["mean_selection_regret"]),
            "std_selection_regret": float(row_best["std_selection_regret"]),
            "mean_selected_minus_sc": float(row_best["mean_selected_minus_sc"]),
            "std_selected_minus_sc": float(row_best["std_selected_minus_sc"]),
            "mean_gsc_selected_ami": float(row_best["mean_gsc_selected_ami"]),
            "mean_gsc_oracle_ami": float(row_best["mean_gsc_oracle_ami"]),
            "mean_sc_ami": float(row_best["mean_sc_ami"]),
            "profile_id": best_profile_id,
        }
    ]

    other_summary_df = pd.read_csv(other_summary_path)
    for _, r in other_summary_df.iterrows():
        rows.append(
            {
                "metric": str(r["metric"]),
                "metric_display": str(r["metric_display"]),
                "metric_optimize": str(r["metric_optimize"]),
                "overall_spearman_aligned": float(r["overall_spearman_aligned"]),
                "overall_pearson_aligned": float(r["overall_pearson_aligned"]),
                "mean_selection_regret": float(r["mean_selection_regret"]),
                "std_selection_regret": float(r["std_selection_regret"]),
                "mean_selected_minus_sc": float(r["mean_selected_minus_sc"]),
                "std_selected_minus_sc": float(r["std_selected_minus_sc"]),
                "mean_gsc_selected_ami": float(r["mean_gsc_selected_ami"]),
                "mean_gsc_oracle_ami": float(r["mean_gsc_oracle_ami"]),
                "mean_sc_ami": float(r["mean_sc_ami"]),
                "profile_id": None,
            }
        )

    df = pd.DataFrame(rows)
    return df


def _bar_plot(df: pd.DataFrame, y: str, yerr: str | None, title: str, ylabel: str, out_path: Path) -> None:
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    if yerr is None:
        ax.bar(x, df[y], color=["#1f77b4", "#d62728", "#2ca02c"][: len(df)])
    else:
        ax.bar(x, df[y], yerr=df[yerr], capsize=4, color=["#1f77b4", "#d62728", "#2ca02c"][: len(df)])
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["metric_display"], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_index(out_dir: Path, df: pd.DataFrame) -> None:
    lines = [
        "# Networks Metric Comparison",
        "",
        "## Objective",
        "",
        "Compare modularity/map-equation against the best Graph-CH filter profile.",
        "",
        "## Summary",
        "",
        "| metric | optimize | spearman_aligned | selection_regret | selected_minus_sc | selected_ami |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['metric_display']} | {r['metric_optimize']} | {r['overall_spearman_aligned']:.4f} | "
            f"{r['mean_selection_regret']:.4f} | {r['mean_selected_minus_sc']:.4f} | {r['mean_gsc_selected_ami']:.4f} |"
        )
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare best Graph-CH profile vs other metrics")
    parser.add_argument(
        "--graphch-analysis-dir",
        default="results/benchmark_networks_graphch_profiles_grid_search/analysis_graphch_profiles",
    )
    parser.add_argument(
        "--other-analysis-dir",
        default="results/benchmark_networks_other_metrics_grid_search/analysis_other_metrics",
    )
    parser.add_argument(
        "--out-dir",
        default="results/benchmark_networks_metric_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _style()

    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    manifest_dir = out_dir / "manifests"
    for d in [tables_dir, fig_dir, manifest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df = build_comparison_df(
        graphch_analysis_dir=Path(args.graphch_analysis_dir),
        other_analysis_dir=Path(args.other_analysis_dir),
    )

    df = df.sort_values(["overall_spearman_aligned", "mean_selection_regret"], ascending=[False, True]).reset_index(drop=True)
    df.to_csv(tables_dir / "best_graphch_vs_others.csv", index=False)

    _bar_plot(
        df,
        y="overall_spearman_aligned",
        yerr=None,
        title="Aligned Spearman correlation with AMI",
        ylabel="Spearman (aligned)",
        out_path=fig_dir / "overall_spearman_comparison.pdf",
    )
    _bar_plot(
        df,
        y="mean_selection_regret",
        yerr="std_selection_regret",
        title="Selection regret comparison",
        ylabel="AMI_oracle - AMI_selected",
        out_path=fig_dir / "selection_regret_comparison.pdf",
    )
    _bar_plot(
        df,
        y="mean_selected_minus_sc",
        yerr="std_selected_minus_sc",
        title="Selected-vs-SC gain comparison",
        ylabel="AMI_selected - AMI_SC",
        out_path=fig_dir / "selected_minus_sc_comparison.pdf",
    )
    _bar_plot(
        df,
        y="mean_gsc_selected_ami",
        yerr=None,
        title="Mean selected AMI comparison",
        ylabel="Mean AMI selected by proxy",
        out_path=fig_dir / "selected_ami_comparison.pdf",
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "graphch_analysis_dir": args.graphch_analysis_dir,
        "other_analysis_dir": args.other_analysis_dir,
        "out_dir": args.out_dir,
    }
    with (manifest_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _write_index(out_dir, df)
    print(f"Saved comparison outputs to: {out_dir}")


if __name__ == "__main__":
    main()
