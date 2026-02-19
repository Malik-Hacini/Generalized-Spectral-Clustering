"""Analyze modularity/map-equation proxy metrics on network benchmarks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


METRIC_SPECS = {
    "modularity": {"display": "Directed Modularity", "optimize": "max"},
    "map_equation": {"display": "Map Equation", "optimize": "min"},
}


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


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan, np.nan
    return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0])


def _dataset_dirs(results_dir: Path, gsc_method: str) -> list[Path]:
    out = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / gsc_method / f"{gsc_method}_all_results.json").exists():
            out.append(d)
    return out


def _parse_measure(entry: dict) -> tuple[float, float]:
    measure = entry.get("measure", [None, {}])
    kwargs = {}
    if isinstance(measure, list) and len(measure) >= 2 and isinstance(measure[1], dict):
        kwargs = measure[1]
    return float(kwargs.get("alpha", np.nan)), float(kwargs.get("t", np.nan))


def _extract_mean(entry: dict, metric: str) -> float:
    return float(entry[metric]["mean"])


def _aligned(metric: str, raw_value: float) -> float:
    if METRIC_SPECS[metric]["optimize"] == "min":
        return -float(raw_value)
    return float(raw_value)


def load_data(results_dir: Path, metrics: list[str], gsc_method: str, sc_method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_rows = []
    selection_rows = []

    for d in _dataset_dirs(results_dir, gsc_method):
        gsc_path = d / gsc_method / f"{gsc_method}_all_results.json"
        sc_path = d / sc_method / f"{sc_method}_all_results.json"
        gsc_all = json.loads(gsc_path.read_text())
        sc_all = json.loads(sc_path.read_text())
        if not gsc_all or not sc_all:
            continue

        sc_ami = float(sc_all[0]["ami"]["mean"])

        for metric in metrics:
            metric_values = []
            ami_values = []
            rows_metric = []

            for idx, entry in enumerate(gsc_all):
                alpha, t = _parse_measure(entry)
                raw = _extract_mean(entry, metric)
                ami = _extract_mean(entry, "ami")
                aligned = _aligned(metric, raw)

                row = {
                    "dataset": d.name,
                    "metric": metric,
                    "metric_display": METRIC_SPECS[metric]["display"],
                    "metric_optimize": METRIC_SPECS[metric]["optimize"],
                    "grid_index": idx,
                    "alpha": alpha,
                    "t": t,
                    "ami": ami,
                    "metric_raw": raw,
                    "metric_aligned": aligned,
                    "sc_ami": sc_ami,
                }
                rows_metric.append(row)
                metric_values.append(raw)
                ami_values.append(ami)

            grid_rows.extend(rows_metric)

            ami_arr = np.asarray(ami_values, dtype=float)
            raw_arr = np.asarray(metric_values, dtype=float)
            aligned_arr = np.asarray([_aligned(metric, v) for v in raw_arr], dtype=float)

            idx_oracle = int(np.argmax(ami_arr))
            idx_selected = int(np.argmax(aligned_arr))
            p_raw, s_raw = _safe_corr(raw_arr, ami_arr)
            p_al, s_al = _safe_corr(aligned_arr, ami_arr)

            selected_row = rows_metric[idx_selected]
            oracle_row = rows_metric[idx_oracle]

            selection_rows.append(
                {
                    "dataset": d.name,
                    "metric": metric,
                    "metric_display": METRIC_SPECS[metric]["display"],
                    "metric_optimize": METRIC_SPECS[metric]["optimize"],
                    "sc_ami": sc_ami,
                    "gsc_ami_selected": float(selected_row["ami"]),
                    "gsc_ami_oracle": float(oracle_row["ami"]),
                    "selection_regret": float(oracle_row["ami"] - selected_row["ami"]),
                    "selected_minus_sc": float(selected_row["ami"] - sc_ami),
                    "oracle_minus_sc": float(oracle_row["ami"] - sc_ami),
                    "dataset_pearson_raw": p_raw,
                    "dataset_spearman_raw": s_raw,
                    "dataset_pearson_aligned": p_al,
                    "dataset_spearman_aligned": s_al,
                    "selected_alpha": float(selected_row["alpha"]),
                    "selected_t": float(selected_row["t"]),
                    "oracle_alpha": float(oracle_row["alpha"]),
                    "oracle_t": float(oracle_row["t"]),
                }
            )

    grid_df = pd.DataFrame(grid_rows)
    selection_df = pd.DataFrame(selection_rows)
    if grid_df.empty or selection_df.empty:
        raise RuntimeError(f"No usable data found in {results_dir}")
    return grid_df, selection_df


def summarize(grid_df: pd.DataFrame, selection_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows = []
    summary_rows = []

    for metric, subset in grid_df.groupby("metric", sort=False):
        p_raw, s_raw = _safe_corr(subset["metric_raw"].to_numpy(dtype=float), subset["ami"].to_numpy(dtype=float))
        p_al, s_al = _safe_corr(subset["metric_aligned"].to_numpy(dtype=float), subset["ami"].to_numpy(dtype=float))
        corr_rows.append(
            {
                "metric": metric,
                "metric_display": str(subset["metric_display"].iloc[0]),
                "metric_optimize": str(subset["metric_optimize"].iloc[0]),
                "n": int(len(subset)),
                "overall_pearson_raw": p_raw,
                "overall_spearman_raw": s_raw,
                "overall_pearson_aligned": p_al,
                "overall_spearman_aligned": s_al,
            }
        )

    corr_df = pd.DataFrame(corr_rows)

    for metric, subset in selection_df.groupby("metric", sort=False):
        corr_row = corr_df[corr_df["metric"] == metric].iloc[0]
        summary_rows.append(
            {
                "metric": metric,
                "metric_display": str(subset["metric_display"].iloc[0]),
                "metric_optimize": str(subset["metric_optimize"].iloc[0]),
                "n_datasets": int(len(subset)),
                "mean_gsc_selected_ami": float(subset["gsc_ami_selected"].mean()),
                "mean_gsc_oracle_ami": float(subset["gsc_ami_oracle"].mean()),
                "mean_sc_ami": float(subset["sc_ami"].mean()),
                "mean_selection_regret": float(subset["selection_regret"].mean()),
                "std_selection_regret": float(subset["selection_regret"].std(ddof=0)),
                "mean_selected_minus_sc": float(subset["selected_minus_sc"].mean()),
                "std_selected_minus_sc": float(subset["selected_minus_sc"].std(ddof=0)),
                "mean_dataset_spearman_aligned": float(subset["dataset_spearman_aligned"].mean()),
                "overall_pearson_aligned": float(corr_row["overall_pearson_aligned"]),
                "overall_spearman_aligned": float(corr_row["overall_spearman_aligned"]),
                "overall_pearson_raw": float(corr_row["overall_pearson_raw"]),
                "overall_spearman_raw": float(corr_row["overall_spearman_raw"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["overall_spearman_aligned", "mean_selection_regret", "mean_gsc_selected_ami"],
        ascending=[False, True, False],
    )
    return corr_df, summary_df


def _plot_metric_scatter(grid_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    data = grid_df[grid_df["metric"] == metric]
    datasets = sorted(data["dataset"].unique().tolist())
    dataset_to_idx = {d: i for i, d in enumerate(datasets)}
    c = np.asarray([dataset_to_idx[d] for d in data["dataset"]], dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    scatter = ax.scatter(data["metric_raw"], data["ami"], c=c, cmap="viridis", s=34, alpha=0.60, edgecolors="none")
    p, s = _safe_corr(data["metric_raw"].to_numpy(dtype=float), data["ami"].to_numpy(dtype=float))
    pa, sa = _safe_corr(data["metric_aligned"].to_numpy(dtype=float), data["ami"].to_numpy(dtype=float))

    if len(data) >= 2:
        coef = np.polyfit(data["metric_raw"], data["ami"], deg=1)
        x = np.linspace(data["metric_raw"].min(), data["metric_raw"].max(), 200)
        y = coef[0] * x + coef[1]
        ax.plot(x, y, color="#111111", linewidth=2.0, label="Linear fit")

    ax.text(
        0.02,
        0.98,
        f"Raw Pearson={p:.3f}\nRaw Spearman={s:.3f}\nAligned Pearson={pa:.3f}\nAligned Spearman={sa:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_ticks(np.arange(len(datasets)))
    cbar.set_ticklabels(datasets)
    cbar.set_label("Dataset")

    display = METRIC_SPECS[metric]["display"]
    ax.set_title(f"GSC grid: AMI vs {display}")
    ax.set_xlabel(display)
    ax.set_ylabel("AMI")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_dataset_bars(selection_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    data = selection_df[selection_df["metric"] == metric].sort_values("dataset")
    x = np.arange(len(data))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    ax.bar(x - width, data["sc_ami"], width=width, label="SC-N", color="#222222")
    ax.bar(x, data["gsc_ami_selected"], width=width, label="GSC-N selected", color="#1f77b4")
    ax.bar(x + width, data["gsc_ami_oracle"], width=width, label="GSC-N oracle", color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels(data["dataset"], rotation=0)
    ax.set_ylabel("AMI")
    ax.set_title(f"{METRIC_SPECS[metric]['display']}: selected vs oracle vs SC")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_summary(summary_df: pd.DataFrame, out_dir: Path) -> None:
    x = np.arange(len(summary_df))

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - 0.18, summary_df["overall_pearson_aligned"], width=0.36, label="Pearson aligned", color="#1f77b4")
    ax.bar(x + 0.18, summary_df["overall_spearman_aligned"], width=0.36, label="Spearman aligned", color="#d62728")
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["metric_display"], rotation=0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Correlation")
    ax.set_title("Proxy-AMI correlation comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "overall_corr_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(
        x,
        summary_df["mean_selection_regret"],
        yerr=summary_df["std_selection_regret"],
        capsize=4,
        color="#2ca02c",
    )
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["metric_display"], rotation=0)
    ax.set_ylabel("AMI_oracle - AMI_selected")
    ax.set_title("Selection regret comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "selection_regret_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(
        x,
        summary_df["mean_selected_minus_sc"],
        yerr=summary_df["std_selected_minus_sc"],
        capsize=4,
        color="#9467bd",
    )
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["metric_display"], rotation=0)
    ax.set_ylabel("AMI_selected - AMI_SC")
    ax.set_title("Selected-vs-SC gain comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "selected_minus_sc_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_index(out_dir: Path, summary_df: pd.DataFrame) -> None:
    lines = [
        "# Networks Other-Metrics Analysis",
        "",
        "## Objective",
        "",
        "Assess modularity and map equation as AMI proxies for GSC model selection.",
        "",
        "## Summary",
        "",
        "| metric | optimize | overall_spearman_aligned | selection_regret | selected_minus_sc |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['metric_display']} | {r['metric_optimize']} | {r['overall_spearman_aligned']:.4f} | "
            f"{r['mean_selection_regret']:.4f} | {r['mean_selected_minus_sc']:.4f} |"
        )
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze modularity/map-equation on network benchmarks")
    parser.add_argument(
        "--results-dir",
        default="results/benchmark_networks_other_metrics_grid_search",
        help="Benchmark directory containing ami+modularity+map_equation",
    )
    parser.add_argument("--metrics", nargs="+", default=["modularity", "map_equation"])
    parser.add_argument("--gsc-method", default="GSC-N")
    parser.add_argument("--sc-method", default="SC-N")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output analysis directory (default: <results-dir>/analysis_other_metrics)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _style()

    for metric in args.metrics:
        if metric not in METRIC_SPECS:
            raise ValueError(f"Unsupported metric '{metric}'. Supported: {sorted(METRIC_SPECS.keys())}")

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "analysis_other_metrics"
    tables_dir = out_dir / "tables"
    fig_per_metric = out_dir / "figures" / "per_metric"
    fig_cross = out_dir / "figures" / "cross_metric"
    manifest_dir = out_dir / "manifests"
    for d in [tables_dir, fig_per_metric, fig_cross, manifest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    grid_df, selection_df = load_data(
        results_dir=results_dir,
        metrics=args.metrics,
        gsc_method=args.gsc_method,
        sc_method=args.sc_method,
    )
    corr_df, summary_df = summarize(grid_df, selection_df)

    grid_df.to_csv(tables_dir / "grid_records.csv", index=False)
    selection_df.to_csv(tables_dir / "selection_per_dataset.csv", index=False)
    corr_df.to_csv(tables_dir / "correlation_overall.csv", index=False)
    summary_df.to_csv(tables_dir / "summary_by_metric.csv", index=False)

    for metric in args.metrics:
        _plot_metric_scatter(grid_df, metric, fig_per_metric / metric / f"ami_vs_{metric}.pdf")
        _plot_metric_dataset_bars(selection_df, metric, fig_per_metric / metric / f"dataset_selection_{metric}.pdf")

    _plot_cross_summary(summary_df, fig_cross)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "metrics": args.metrics,
        "gsc_method": args.gsc_method,
        "sc_method": args.sc_method,
    }
    with (manifest_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _write_index(out_dir, summary_df)
    print(f"Saved other-metrics analysis to: {out_dir}")


if __name__ == "__main__":
    main()
