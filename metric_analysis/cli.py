"""CLI entrypoint for multi-metric DSBM proxy-correlation analysis."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from metric_analysis.io import load_grid_and_baselines
from metric_analysis.plots import (
    plot_cross_metric_overall_corr,
    plot_cross_metric_regret,
    plot_cross_metric_selected_gain,
    plot_metric_ami_vs_proxy,
    plot_metric_corr_by_gamma,
    plot_metric_gamma_vs_ami,
    plot_metric_regret_by_gamma,
    style_matplotlib,
)
from metric_analysis.selection import compute_dataset_selection
from metric_analysis.specs import resolve_metric_specs
from metric_analysis.stats import compute_grid_correlations, summarize_metrics


def _apply_profile_filter(grid_long_df: pd.DataFrame, profile_id: str | None) -> tuple[pd.DataFrame, str | None]:
    """Filter profile-swept grids to a single profile when requested.

    Returns
    -------
    (filtered_df, selected_profile_id)
        selected_profile_id is None when no profile dimension exists.
    """
    if "profile_id" not in grid_long_df.columns:
        return grid_long_df, None

    present = sorted(
        {
            str(v)
            for v in grid_long_df["profile_id"].dropna().unique().tolist()
            if str(v).strip() != ""
        }
    )

    if not present:
        return grid_long_df, None

    if profile_id is not None:
        if profile_id not in present:
            raise ValueError(
                f"Requested profile_id '{profile_id}' not found. Available: {present}"
            )
        filtered = grid_long_df[grid_long_df["profile_id"] == profile_id].copy()
        return filtered, profile_id

    # If profile-swept Graph-CH is present and user did not choose a profile,
    # fail loudly to avoid misleading pooled correlations.
    graph_ch_rows = grid_long_df[grid_long_df["metric"] == "graph_ch"]
    graph_ch_profiles = sorted(
        {
            str(v)
            for v in graph_ch_rows["profile_id"].dropna().unique().tolist()
            if str(v).strip() != ""
        }
    )

    if len(graph_ch_profiles) > 1:
        raise RuntimeError(
            "Detected multiple graph_ch profiles in results. "
            "Please set --profile-id to analyze a single filter profile. "
            f"Available profiles: {graph_ch_profiles}"
        )

    selected = graph_ch_profiles[0] if graph_ch_profiles else present[0]
    filtered = grid_long_df[grid_long_df["profile_id"] == selected].copy()
    return filtered, selected


def _write_index(
    out_dir: Path,
    summary_df: pd.DataFrame,
    metric_names: list[str],
    results_dir: Path,
    selected_profile_id: str | None,
) -> None:
    ranked = summary_df.sort_values(
        ["mean_selected_minus_sc", "overall_spearman_aligned", "mean_selection_regret"],
        ascending=[False, False, True],
    )

    lines = [
        "# DSBM Proxy-Metric Analysis",
        "",
        "## Objective",
        "",
        "Assess proxy metrics for GSC model selection on DSBM benchmarks.",
        "A good proxy should both correlate with AMI on the GSC grid and",
        "yield low-regret AMI when used to pick (alpha, t).",
        "",
        "## Inputs",
        "",
        f"- Results directory: `{results_dir}`",
        f"- Proxy metrics: `{', '.join(metric_names)}`",
        f"- Profile filter: `{selected_profile_id}`" if selected_profile_id else "- Profile filter: none",
        "",
        "## Output structure",
        "",
        "- `tables/`: machine-readable CSV outputs",
        "- `figures/per_metric/<metric>/`: metric-specific PDF figures",
        "- `figures/cross_metric/`: side-by-side metric comparisons",
        "",
        "## Metric ranking",
        "",
        "| metric | objective | selected_minus_sc | selection_regret | overall_spearman_aligned |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in ranked.iterrows():
        lines.append(
            f"| {row['metric_display']} ({row['metric']}) | "
            f"{row['metric_optimize']} | "
            f"{row['mean_selected_minus_sc']:.4f} | "
            f"{row['mean_selection_regret']:.4f} | "
            f"{row['overall_spearman_aligned']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Suggested reading order",
            "",
            "1. `figures/cross_metric/overall_corr_comparison.pdf`",
            "2. `figures/cross_metric/selection_regret_comparison.pdf`",
            "3. `figures/cross_metric/gsc_selected_minus_sc_comparison.pdf`",
            "4. Metric-specific figures in `figures/per_metric/<metric>/`",
        ]
    )

    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DSBM correlations for multiple proxy metrics")
    parser.add_argument(
        "--results-dir",
        default="results/benchmark_dsbm_grid_search",
        help="Grid-search results directory (contains dsbm_gamma* folders)",
    )
    parser.add_argument(
        "--proxy-metrics",
        nargs="+",
        default=["graph_ch", "modularity", "map_equation"],
        help="Proxy metrics to analyze",
    )
    parser.add_argument("--gsc-method", default="GSC-N", help="GSC method folder name")
    parser.add_argument("--sc-method", default="SC-N", help="SC baseline method name")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <results-dir>/metric_analysis)",
    )
    parser.add_argument(
        "--profile-id",
        default=None,
        help="Optional profile_id filter for profile-swept runs (e.g. delta_k01)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "metric_analysis"
    manifests_dir = out_dir / "manifests"
    tables_dir = out_dir / "tables"
    per_metric_fig_dir = out_dir / "figures" / "per_metric"
    cross_fig_dir = out_dir / "figures" / "cross_metric"

    for directory in [manifests_dir, tables_dir, per_metric_fig_dir, cross_fig_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    style_matplotlib()
    metric_specs = resolve_metric_specs(args.proxy_metrics)

    grid_long_df, baseline_df = load_grid_and_baselines(
        results_dir=results_dir,
        metric_specs=metric_specs,
        gsc_method=args.gsc_method,
        sc_method=args.sc_method,
    )

    grid_long_df, selected_profile_id = _apply_profile_filter(grid_long_df, args.profile_id)

    if selected_profile_id:
        baseline_keep = set(grid_long_df["dataset"].unique().tolist())
        baseline_df = baseline_df[baseline_df["dataset"].isin(baseline_keep)].copy()

    selection_df = compute_dataset_selection(grid_long_df=grid_long_df, baseline_df=baseline_df)
    overall_corr_df, corr_by_gamma_df, corr_by_dataset_df = compute_grid_correlations(grid_long_df)
    summary_df = summarize_metrics(
        selection_df=selection_df,
        overall_corr_df=overall_corr_df,
        dataset_corr_df=corr_by_dataset_df,
    )

    grid_long_df.to_csv(tables_dir / "grid_records_long.csv", index=False)
    baseline_df.to_csv(tables_dir / "baseline_sc_ami.csv", index=False)
    selection_df.to_csv(tables_dir / "selection_per_dataset.csv", index=False)
    overall_corr_df.to_csv(tables_dir / "correlation_overall.csv", index=False)
    corr_by_gamma_df.to_csv(tables_dir / "correlation_by_gamma.csv", index=False)
    corr_by_dataset_df.to_csv(tables_dir / "correlation_by_dataset.csv", index=False)
    summary_df.to_csv(tables_dir / "summary_by_metric.csv", index=False)

    for metric_name in summary_df["metric"].tolist():
        metric_grid = grid_long_df[grid_long_df["metric"] == metric_name]
        metric_selection = selection_df[selection_df["metric"] == metric_name]
        metric_corr_gamma = corr_by_gamma_df[corr_by_gamma_df["metric"] == metric_name]
        metric_overall_row = overall_corr_df[overall_corr_df["metric"] == metric_name].iloc[0]

        metric_dir = per_metric_fig_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)

        plot_metric_ami_vs_proxy(metric_grid, metric_overall_row, metric_dir / f"ami_vs_{metric_name}.pdf")
        plot_metric_corr_by_gamma(metric_corr_gamma, metric_dir / f"corr_by_gamma_{metric_name}.pdf")
        plot_metric_gamma_vs_ami(metric_selection, metric_dir / f"gamma_vs_ami_selected_by_{metric_name}.pdf")
        plot_metric_regret_by_gamma(metric_selection, metric_dir / f"selection_regret_by_gamma_{metric_name}.pdf")

    plot_cross_metric_overall_corr(summary_df, cross_fig_dir / "overall_corr_comparison.pdf")
    plot_cross_metric_regret(summary_df, cross_fig_dir / "selection_regret_comparison.pdf")
    plot_cross_metric_selected_gain(summary_df, cross_fig_dir / "gsc_selected_minus_sc_comparison.pdf")

    manifest_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "out_dir": str(out_dir),
        "proxy_metrics": [spec.name for spec in metric_specs],
        "gsc_method": args.gsc_method,
        "sc_method": args.sc_method,
        "profile_id": selected_profile_id,
        "n_grid_rows": int(len(grid_long_df)),
        "n_selection_rows": int(len(selection_df)),
    }
    with (manifests_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    _write_index(
        out_dir=out_dir,
        summary_df=summary_df,
        metric_names=[spec.name for spec in metric_specs],
        results_dir=results_dir,
        selected_profile_id=selected_profile_id,
    )

    print(f"Saved metric analysis to: {out_dir}")
    print("  - tables/summary_by_metric.csv")
    print("  - figures/cross_metric/*.pdf")
    print("  - figures/per_metric/<metric>/*.pdf")
    print("  - index.md")


if __name__ == "__main__":
    main()
