"""
Plot AMI scores vs degree imbalance ratio for degree-imbalance DSBM benchmark.

Creates a line plot showing mean AMI +/- std for each clustering method
as a function of the out-probability ratio p_low / p_high.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_input_path(path_value: str | Path) -> Path:
    """Resolve paths robustly for runs launched outside the repository root."""
    path = Path(path_value)
    if path.exists() or path.is_absolute():
        return path

    repo_root_candidate = Path(__file__).resolve().parents[1]
    repo_relative = repo_root_candidate / path
    if repo_relative.exists():
        return repo_relative

    return path


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
    results_dir = _resolve_input_path(results_path)

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


def plot_degree_imbalance_results(df: pd.DataFrame, output_file: str = None):
    """
    Create line plot of AMI vs degree imbalance ratio (p_low / p_high).

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame from load_degree_imbalance_results
    output_file : str, optional
        Path to save the plot
    """
    summary = (
        df.groupby(["method", "ratio"])  # aggregate across seeds
        .agg({"ami": ["mean", "std", "count"]})
        .reset_index()
    )
    summary.columns = ["method", "ratio", "ami_mean", "ami_std", "n_seeds"]
    summary = summary.sort_values("ratio")

    method_order = ["SC-UN", "SC-N", "DSC+", "GSC-UN", "GSC-N"]
    method_styles = {
        "SC-UN": {"color": "#FF8C69", "linestyle": ":", "marker": "D", "label": "SC-UN"},
        "SC-N": {"color": "#FF6347", "linestyle": "--", "marker": "o", "label": "SC-N"},
        "DSC+": {"color": "#27A727", "linestyle": "-.", "marker": "^", "label": "DSC+"},
        "GSC-UN": {"color": "#4C9AFF", "linestyle": "--", "marker": "P", "label": "GSC-UN"},
        "GSC-N": {"color": "#072AC8", "linestyle": "-", "marker": "s", "label": "GSC-N"},
    }

    plt.figure()

    for method in method_order:
        method_data = summary[summary["method"] == method]
        if method_data.empty:
            continue

        style = method_styles.get(method, {})
        ratio = method_data["ratio"].values
        ami_mean = method_data["ami_mean"].values
        ami_std = np.nan_to_num(method_data["ami_std"].values, nan=0.0)

        plt.plot(
            ratio,
            ami_mean,
            label=style.get("label", method),
            color=style.get("color", None),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            markersize=6,
            linewidth=2,
            alpha=1,
        )

        plt.fill_between(
            ratio,
            ami_mean - ami_std,
            ami_mean + ami_std,
            color=style.get("color", None),
            alpha=0.2,
        )

    plt.xlabel(r"Degree Imbalance Ratio ($p_{\mathrm{low}} / p_{\mathrm{high}}$)", fontsize=12)
    plt.ylabel("AMI Score", fontsize=12)
    plt.legend(loc="best", fontsize=10, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics of the results."""
    print("=" * 80)
    print("DEGREE-IMBALANCE BENCHMARK SUMMARY")
    print("=" * 80)
    print()

    overall = df.groupby("method").agg({"ami": ["mean", "std", "min", "max", "count"]}).round(4)
    print("Overall AMI Statistics by Method:")
    print(overall)
    print()

    by_ratio = df.groupby(["ratio", "method"])["ami"].agg(["mean", "std"]).unstack("method").round(4)
    print("AMI by Degree Imbalance Ratio:")
    print(by_ratio)
    print()

    best_by_ratio = df.groupby(["ratio", "method"])["ami"].mean().unstack("method")
    print("Best Method by Degree Imbalance Ratio:")
    for ratio in sorted(best_by_ratio.index):
        best_method = best_by_ratio.loc[ratio].idxmax()
        best_score = best_by_ratio.loc[ratio].max()
        print(f"  Ratio {ratio:.4f}: {best_method} (AMI = {best_score:.4f})")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot results from degree-imbalance benchmark")
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/benchmark_degree_imbalance_grid_search",
        help="Path to degree-imbalance grid-search results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Output directory for plots (default: figures/)",
    )
    parser.add_argument("--show-stats", action="store_true", help="Print summary statistics")

    args = parser.parse_args()

    output_dir = _resolve_input_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.results_path}")
    df = load_degree_imbalance_results(args.results_path)

    if df.empty:
        print("No results found!")
        return

    print(f"Loaded {len(df)} result entries")
    print(f"Methods: {sorted(df['method'].unique())}")

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
        plot_degree_imbalance_results(df_setting, str(output_file))
        print(f"Saved plot to: {output_file}")

    print(f"\n{'=' * 60}")
    print(f"All plots saved to: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
