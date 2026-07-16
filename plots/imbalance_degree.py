"""Plot AMI scores vs degree-imbalance ratio for the DSBM benchmark."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from plots.common import (
    configure_paper_style,
    plot_method_lines,
    project_path,
    resolve_output_dir,
    summarize_mean_std,
    validate_selection,
)


def _parse_prob_token(token: str) -> float:
    """Parse probability tokens formatted as e.g. 0p0133 -> 0.0133."""
    return float(token.replace("p", "."))


def load_degree_imbalance_results(results_path: str | Path):
    """
    Load results from degree-imbalance benchmark.

    Parameters
    ----------
    results_path : str or Path
        Path to the benchmark_degree_imbalance results directory

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        method, block_sizes, p_intra, p_high, p_low, ratio, seed, ami
    """
    results_dir = project_path(results_path)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    best_result_files = sorted(results_dir.glob("*/*/*_best_results.json"))

    if not best_result_files:
        raise ValueError(f"No *_best_results.json files found in {results_dir}")

    rows = []
    unmatched_target_names = []
    name_patterns = [
        re.compile(
            r"^disbm_degimbal_b([0-9-]+)_pintra([0-9p]+)_phigh([0-9p]+)_plow([0-9p]+)_seed(\d+)$"
        ),
        re.compile(
            r"^disbm_degree_imbalance_b([0-9-]+)_pintra([0-9p]+)_phigh([0-9p]+)_plow([0-9p]+)_seed(\d+)$"
        ),
        re.compile(
            r"^dcdisbm_degimbal_b([0-9-]+)_pintra([0-9p]+)_pinter([0-9p]+)_highscale([0-9p]+)_lowscale([0-9p]+)_seed(\d+)$"
        ),
    ]

    for best_file in best_result_files:
        method_name = best_file.parent.name
        dataset_name = best_file.parent.parent.name

        match = None
        for pattern in name_patterns:
            match = pattern.match(dataset_name)
            if match is not None:
                break
        if not match:
            if dataset_name.startswith(("disbm", "dcdisbm")):
                unmatched_target_names.append(dataset_name)
            continue

        block_sizes_token = match.group(1)
        p_intra_token = match.group(2)
        if "_pinter" in dataset_name and "_highscale" in dataset_name:
            # dcdisbm format: keep column names for backward compatibility with plotting code.
            p_high_token = match.group(4)
            p_low_token = match.group(5)
            seed = int(match.group(6))
        else:
            p_high_token = match.group(3)
            p_low_token = match.group(4)
            seed = int(match.group(5))

        block_sizes = tuple(int(v) for v in block_sizes_token.split("-"))
        p_intra = _parse_prob_token(p_intra_token)
        p_high = _parse_prob_token(p_high_token)
        p_low = _parse_prob_token(p_low_token)

        if p_high == 0:
            print(f"Warning: p_high=0 in dataset {dataset_name}, skipping")
            continue

        ratio = p_low / p_high

        with open(best_file, "r") as f:
            best_results = json.load(f)

        # Prefer AMI value selected by graph_ch optimization when present.
        ami_value = None
        if "graph_ch" in best_results:
            optimized_data = best_results["graph_ch"]
            if "ami" in optimized_data and "mean" in optimized_data["ami"]:
                ami_value = optimized_data["ami"]["mean"]
        elif "ami" in best_results:
            optimized_data = best_results["ami"]
            if "ami" in optimized_data and "mean" in optimized_data["ami"]:
                ami_value = optimized_data["ami"]["mean"]

        if ami_value is not None:
            rows.append(
                {
                    "method": method_name,
                    "block_sizes": block_sizes,
                    "p_intra": p_intra,
                    "p_high": p_high,
                    "p_low": p_low,
                    "ratio": ratio,
                    "seed": seed,
                    "ami": ami_value,
                }
            )

    if unmatched_target_names:
        unique_unmatched = sorted(set(unmatched_target_names))
        print(
            f"Warning: Could not parse {len(unique_unmatched)} target dataset names. "
            f"First example: {unique_unmatched[0]}"
        )

    return pd.DataFrame(rows)


def plot_degree_imbalance_results(df: pd.DataFrame, output_file: Path):
    summary = summarize_mean_std(df, ["method", "ratio"], "ami").sort_values("ratio")
    fig, ax = plt.subplots()
    plot_method_lines(ax, summary, "ratio", "ami_mean", y_std_col="ami_std")
    ax.set_xlabel(
        r"Degree Imbalance Ratio ($p_{\mathrm{low}} / p_{\mathrm{high}}$)", fontsize=12
    )
    ax.set_ylabel("AMI Score", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_file}")


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics of the results."""
    print("=" * 80)
    print("DEGREE-IMBALANCE BENCHMARK SUMMARY")
    print("=" * 80)
    print()

    overall = (
        df.groupby("method")
        .agg({"ami": ["mean", "std", "min", "max", "count"]})
        .round(4)
    )
    print("Overall AMI Statistics by Method:")
    print(overall)
    print()

    by_ratio = (
        df.groupby(["ratio", "method"])["ami"]
        .agg(["mean", "std"])
        .unstack("method")
        .round(4)
    )
    print("AMI by Degree Imbalance Ratio:")
    print(by_ratio)
    print()

    best_by_ratio = df.groupby(["ratio", "method"])["ami"].mean().unstack("method")
    print("Best Method by Degree Imbalance Ratio:")
    for ratio in sorted(best_by_ratio.index):
        best_method = best_by_ratio.loc[ratio].idxmax()
        best_score = best_by_ratio.loc[ratio].max()
        print(f"  Ratio {ratio:.4f}: {best_method} (AMI = {best_score:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot results from degree-imbalance benchmark"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/benchmark_degree_imbalance_grid_search",
        help="Path to degree-imbalance grid-search results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to plots/imbalance/<experiment_name>/.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to plot (default: all methods)",
    )
    parser.add_argument(
        "--show-stats", action="store_true", help="Print summary statistics"
    )

    args = parser.parse_args()
    configure_paper_style(plt)

    output_dir = resolve_output_dir(args.output_dir, "imbalance", args.results_dir)

    print(f"Loading results from: {args.results_dir}")
    df = load_degree_imbalance_results(args.results_dir)

    if df.empty:
        print("No results found!")
        return

    print(f"Loaded {len(df)} result entries")
    print(f"Methods: {sorted(df['method'].unique())}")

    selected_methods = validate_selection(
        sorted(df["method"].unique()), args.methods, "methods"
    )
    df = df[df["method"].isin(selected_methods)].copy()

    settings = sorted(
        df[["block_sizes", "p_intra", "p_high"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    print(f"Fixed settings found: {len(settings)}")

    for block_sizes, p_intra, p_high in settings:
        df_setting = df[
            (df["block_sizes"] == block_sizes)
            & (df["p_intra"] == p_intra)
            & (df["p_high"] == p_high)
        ]

        block_token = "-".join(str(v) for v in block_sizes)

        print(f"\n{'=' * 60}")
        print(
            "Processing setting: "
            f"blocks={block_sizes}, p_intra={p_intra:.4f}, p_high={p_high:.4f}"
        )
        print(f"{'=' * 60}")
        print(f"Seeds: {len(df_setting['seed'].unique())}")
        print(f"Ratios: {[round(v, 4) for v in sorted(df_setting['ratio'].unique())]}")

        if args.show_stats:
            print()
            print_summary_statistics(df_setting)
            print()

        output_file = output_dir / (
            f"degree_imbalance_b{block_token}_pintra{p_intra:.4f}_phigh{p_high:.4f}.pdf"
        )
        plot_degree_imbalance_results(df_setting, output_file)
        print(f"Saved plot to: {output_file}")

    print(f"\n{'=' * 60}")
    print(f"All plots saved to: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
