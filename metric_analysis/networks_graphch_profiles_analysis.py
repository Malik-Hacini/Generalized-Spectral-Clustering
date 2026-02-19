"""Analyze Graph-CH filter profiles on network benchmark datasets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


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


def _parse_profile(entry: dict) -> tuple[str, str, int]:
    metric_params = entry.get("metric_params", None)
    if not isinstance(metric_params, dict):
        raise RuntimeError("Graph-CH profile entry has no metric_params dict")

    profile_id = str(metric_params.get("profile_id", ""))
    profile_family = str(metric_params.get("profile_family", ""))
    profile_scale = int(metric_params.get("profile_scale", -1))
    if not profile_id:
        raise RuntimeError("Graph-CH profile entry missing profile_id")
    return profile_id, profile_family, profile_scale


def _extract_mean(entry: dict, metric: str) -> float:
    payload = entry[metric]
    return float(payload["mean"])


def load_graphch_profile_data(results_dir: Path, gsc_method: str, sc_method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_rows = []
    dataset_profile_rows = []

    for d in _dataset_dirs(results_dir, gsc_method):
        gsc_path = d / gsc_method / f"{gsc_method}_all_results.json"
        sc_path = d / sc_method / f"{sc_method}_all_results.json"

        gsc_all = json.loads(gsc_path.read_text())
        sc_all = json.loads(sc_path.read_text())
        if not gsc_all or not sc_all:
            continue

        sc_ami = float(sc_all[0]["ami"]["mean"])

        for entry in gsc_all:
            profile_id, profile_family, profile_scale = _parse_profile(entry)
            alpha, t = _parse_measure(entry)
            grid_rows.append(
                {
                    "dataset": d.name,
                    "profile_id": profile_id,
                    "profile_family": profile_family,
                    "profile_scale": profile_scale,
                    "alpha": alpha,
                    "t": t,
                    "ami": _extract_mean(entry, "ami"),
                    "graph_ch": _extract_mean(entry, "graph_ch"),
                    "sc_ami": sc_ami,
                }
            )

        df_dataset = pd.DataFrame([r for r in grid_rows if r["dataset"] == d.name])
        for profile_id, subset in df_dataset.groupby("profile_id", sort=False):
            subset = subset.sort_values(["alpha", "t"]).reset_index(drop=True)
            ami = subset["ami"].to_numpy(dtype=float)
            graph_ch = subset["graph_ch"].to_numpy(dtype=float)

            pearson, spearman = _safe_corr(graph_ch, ami)
            idx_oracle = int(np.argmax(ami))
            idx_selected = int(np.argmax(graph_ch))

            row_oracle = subset.iloc[idx_oracle]
            row_selected = subset.iloc[idx_selected]

            dataset_profile_rows.append(
                {
                    "dataset": d.name,
                    "profile_id": profile_id,
                    "profile_family": str(row_selected["profile_family"]),
                    "profile_scale": int(row_selected["profile_scale"]),
                    "sc_ami": sc_ami,
                    "gsc_ami_oracle": float(row_oracle["ami"]),
                    "gsc_ami_selected": float(row_selected["ami"]),
                    "selection_regret": float(row_oracle["ami"] - row_selected["ami"]),
                    "selected_minus_sc": float(row_selected["ami"] - sc_ami),
                    "oracle_minus_sc": float(row_oracle["ami"] - sc_ami),
                    "dataset_grid_pearson": pearson,
                    "dataset_grid_spearman": spearman,
                    "selected_alpha": float(row_selected["alpha"]),
                    "selected_t": float(row_selected["t"]),
                    "oracle_alpha": float(row_oracle["alpha"]),
                    "oracle_t": float(row_oracle["t"]),
                }
            )

    grid_df = pd.DataFrame(grid_rows)
    dataset_profile_df = pd.DataFrame(dataset_profile_rows)
    if grid_df.empty or dataset_profile_df.empty:
        raise RuntimeError(f"No graph_ch profile data found in {results_dir}")
    return grid_df, dataset_profile_df


def summarize_profiles(grid_df: pd.DataFrame, dataset_profile_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile_id, subset in dataset_profile_df.groupby("profile_id", sort=False):
        family = str(subset["profile_family"].iloc[0])
        scale = int(subset["profile_scale"].iloc[0])
        grid_subset = grid_df[grid_df["profile_id"] == profile_id]

        p_all, s_all = _safe_corr(
            grid_subset["graph_ch"].to_numpy(dtype=float),
            grid_subset["ami"].to_numpy(dtype=float),
        )

        rows.append(
            {
                "profile_id": profile_id,
                "profile_family": family,
                "profile_scale": scale,
                "overall_pearson": p_all,
                "overall_spearman": s_all,
                "mean_dataset_pearson": float(subset["dataset_grid_pearson"].mean()),
                "mean_dataset_spearman": float(subset["dataset_grid_spearman"].mean()),
                "mean_gsc_selected_ami": float(subset["gsc_ami_selected"].mean()),
                "mean_gsc_oracle_ami": float(subset["gsc_ami_oracle"].mean()),
                "mean_sc_ami": float(subset["sc_ami"].mean()),
                "mean_selected_minus_sc": float(subset["selected_minus_sc"].mean()),
                "std_selected_minus_sc": float(subset["selected_minus_sc"].std(ddof=0)),
                "mean_selection_regret": float(subset["selection_regret"].mean()),
                "std_selection_regret": float(subset["selection_regret"].std(ddof=0)),
                "n_datasets": int(len(subset)),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["overall_spearman", "mean_selection_regret", "mean_gsc_selected_ami"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    return summary


def _plot_family_curves(summary_df: pd.DataFrame, y_col_1: str, y_col_2: str | None, title: str, y_label: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    families = ["delta_k", "prefix_k"]
    colors = ["#1f77b4", "#d62728"]

    for ax, family in zip(axes, families):
        data = summary_df[summary_df["profile_family"] == family].sort_values("profile_scale")
        ax.plot(data["profile_scale"], data[y_col_1], marker="o", linewidth=2.0, color=colors[0], label=y_col_1)
        if y_col_2 is not None:
            ax.plot(data["profile_scale"], data[y_col_2], marker="s", linewidth=2.0, color=colors[1], label=y_col_2)
        ax.set_title(f"{family}")
        ax.set_xlabel("Scale")
        if y_col_2 is not None:
            ax.legend(frameon=False)
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)

    axes[0].set_ylabel(y_label)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_best_profile_dataset_bars(dataset_profile_df: pd.DataFrame, best_profile_id: str, out_path: Path) -> None:
    data = dataset_profile_df[dataset_profile_df["profile_id"] == best_profile_id].sort_values("dataset")
    x = np.arange(len(data))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    ax.bar(x - width, data["sc_ami"], width=width, label="SC-N", color="#222222")
    ax.bar(x, data["gsc_ami_selected"], width=width, label="GSC-N selected by Graph-CH", color="#1f77b4")
    ax.bar(x + width, data["gsc_ami_oracle"], width=width, label="GSC-N oracle", color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels(data["dataset"], rotation=0)
    ax.set_ylabel("AMI")
    ax.set_title(f"Best Graph-CH profile ({best_profile_id}) across datasets")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_best_profile_scatter(grid_df: pd.DataFrame, best_profile_id: str, out_path: Path) -> None:
    data = grid_df[grid_df["profile_id"] == best_profile_id].copy()
    datasets = sorted(data["dataset"].unique().tolist())
    dataset_to_idx = {d: i for i, d in enumerate(datasets)}
    color_values = np.asarray([dataset_to_idx[d] for d in data["dataset"]], dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    scatter = ax.scatter(
        data["graph_ch"],
        data["ami"],
        c=color_values,
        cmap="viridis",
        s=30,
        alpha=0.60,
        edgecolors="none",
    )

    if len(data) >= 2:
        coeffs = np.polyfit(data["graph_ch"], data["ami"], deg=1)
        x = np.linspace(data["graph_ch"].min(), data["graph_ch"].max(), 200)
        y = coeffs[0] * x + coeffs[1]
        ax.plot(x, y, color="#111111", linewidth=2.0, label="Linear fit")

    p, s = _safe_corr(data["graph_ch"].to_numpy(dtype=float), data["ami"].to_numpy(dtype=float))
    ax.text(
        0.02,
        0.98,
        f"Pearson r={p:.3f}\nSpearman rho={s:.3f}\nn={len(data)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_ticks(np.arange(len(datasets)))
    cbar.set_ticklabels(datasets)
    cbar.set_label("Dataset")

    ax.set_title(f"Best profile ({best_profile_id}): AMI vs Graph-CH")
    ax.set_xlabel("Graph-CH")
    ax.set_ylabel("AMI")
    ax.legend(frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_index(out_dir: Path, summary_df: pd.DataFrame, best_profile_id: str) -> None:
    top = summary_df.head(6)
    lines = [
        "# Networks Graph-CH Profile Analysis",
        "",
        "## Objective",
        "",
        "Identify the Graph-CH filter profile that best tracks AMI and yields",
        "low-regret GSC model selection across network datasets.",
        "",
        f"## Best profile: `{best_profile_id}`",
        "",
        "## Top profiles",
        "",
        "| profile_id | family | scale | overall_spearman | selection_regret | selected_minus_sc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| {r['profile_id']} | {r['profile_family']} | {int(r['profile_scale'])} | "
            f"{r['overall_spearman']:.4f} | {r['mean_selection_regret']:.4f} | {r['mean_selected_minus_sc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Reading order",
            "",
            "1. `figures/corr_vs_scale.pdf`",
            "2. `figures/regret_vs_scale.pdf`",
            "3. `figures/selected_minus_sc_vs_scale.pdf`",
            "4. `figures/best_profile_dataset_bars.pdf`",
            "5. `figures/best_profile_ami_vs_graphch.pdf`",
        ]
    )
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Graph-CH profile sweep on network datasets")
    parser.add_argument(
        "--results-dir",
        default="results/benchmark_networks_graphch_profiles_grid_search",
        help="Benchmark directory containing graph_ch profile sweep",
    )
    parser.add_argument("--gsc-method", default="GSC-N")
    parser.add_argument("--sc-method", default="SC-N")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output analysis directory (default: <results-dir>/analysis_graphch_profiles)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _style()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "analysis_graphch_profiles"
    tables_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    manifest_dir = out_dir / "manifests"

    for d in [tables_dir, fig_dir, manifest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    grid_df, dataset_profile_df = load_graphch_profile_data(
        results_dir=results_dir,
        gsc_method=args.gsc_method,
        sc_method=args.sc_method,
    )
    summary_df = summarize_profiles(grid_df, dataset_profile_df)

    best = summary_df.iloc[0]
    best_profile_id = str(best["profile_id"])

    grid_df.to_csv(tables_dir / "grid_records_graphch.csv", index=False)
    dataset_profile_df.to_csv(tables_dir / "dataset_profile_metrics.csv", index=False)
    summary_df.to_csv(tables_dir / "profile_summary.csv", index=False)

    _plot_family_curves(
        summary_df,
        y_col_1="overall_pearson",
        y_col_2="overall_spearman",
        title="Graph-CH profile quality: AMI correlation",
        y_label="Correlation with AMI",
        out_path=fig_dir / "corr_vs_scale.pdf",
    )
    _plot_family_curves(
        summary_df,
        y_col_1="mean_selection_regret",
        y_col_2=None,
        title="Graph-CH profile quality: selection regret",
        y_label="AMI_oracle - AMI_selected",
        out_path=fig_dir / "regret_vs_scale.pdf",
    )
    _plot_family_curves(
        summary_df,
        y_col_1="mean_selected_minus_sc",
        y_col_2=None,
        title="Graph-CH profile quality: selected-vs-SC gain",
        y_label="AMI_selected - AMI_SC",
        out_path=fig_dir / "selected_minus_sc_vs_scale.pdf",
    )
    _plot_best_profile_dataset_bars(dataset_profile_df, best_profile_id, fig_dir / "best_profile_dataset_bars.pdf")
    _plot_best_profile_scatter(grid_df, best_profile_id, fig_dir / "best_profile_ami_vs_graphch.pdf")

    best_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "best_profile": {
            "profile_id": best_profile_id,
            "profile_family": str(best["profile_family"]),
            "profile_scale": int(best["profile_scale"]),
            "overall_pearson": float(best["overall_pearson"]),
            "overall_spearman": float(best["overall_spearman"]),
            "mean_selection_regret": float(best["mean_selection_regret"]),
            "mean_selected_minus_sc": float(best["mean_selected_minus_sc"]),
            "mean_gsc_selected_ami": float(best["mean_gsc_selected_ami"]),
        },
    }
    with (manifest_dir / "best_profile.json").open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    _write_index(out_dir, summary_df, best_profile_id)

    print(f"Saved graph_ch profile analysis to: {out_dir}")
    print(f"Best profile: {best_profile_id}")


if __name__ == "__main__":
    main()
