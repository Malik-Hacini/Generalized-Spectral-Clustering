"""Plot AMI scores vs imbalance ratio for the grid-imbalance benchmark."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.common import project_path, resolve_output_dir
from plots.method_style import ordered_methods, style_for_method

# Configurable default methods to plot (None = all methods)
DEFAULT_METHODS_TO_PLOT = [
    "GSC-N",
    "SC-N",
    "DSC+",
]


def load_grid_imbalance_results(results_path: str | Path):
    results_dir = project_path(results_path)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    best_result_files = sorted(results_dir.glob("*/*/*_best_results.json"))
    if not best_result_files:
        raise ValueError(f"No best_results.json files found in {results_dir}")

    rows = []
    for best_file in best_result_files:
        method_name = best_file.parent.name
        dataset_name = best_file.parent.parent.name

        square_match = re.match(r"grid_(\d+)x(\d+)_high(\d+)_low(\d+)_seed(\d+)$", dataset_name)
        tuple_match = re.match(
            r"grid_\((\d+),\s*(\d+)\)x\((\d+),\s*(\d+)\)_high(\d+)_low(\d+)_seed(\d+)$",
            dataset_name,
        )

        if square_match:
            grid_rows = int(square_match.group(1))
            grid_cols = int(square_match.group(2))
            n_high = int(square_match.group(3))
            n_low = int(square_match.group(4))
            seed = int(square_match.group(5))
        elif tuple_match:
            grid_rows = int(tuple_match.group(1))
            grid_cols = int(tuple_match.group(2))
            n_high = int(tuple_match.group(5))
            n_low = int(tuple_match.group(6))
            seed = int(tuple_match.group(7))
        else:
            print(f"Warning: Could not parse dataset name: {dataset_name}")
            continue

        ratio = n_low / n_high
        best_results = json.loads(best_file.read_text())

        ami_value = None
        if "ch" in best_results:
            optimized_data = best_results["ch"]
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
                    "grid_rows": grid_rows,
                    "grid_cols": grid_cols,
                    "n_high": n_high,
                    "n_low": n_low,
                    "ratio": ratio,
                    "seed": seed,
                    "ami": ami_value,
                }
            )

    return pd.DataFrame(rows)


def plot_imbalance_results(df: pd.DataFrame, output_file: Path) -> None:
    summary = df.groupby(["method", "ratio"]).agg({"ami": ["mean", "std"]}).reset_index()
    summary.columns = ["method", "ratio", "ami_mean", "ami_std"]
    summary = summary.sort_values("ratio")

    method_order = ordered_methods(summary["method"].unique().tolist())

    plt.figure()
    for method in method_order:
        method_data = summary[summary["method"] == method]
        if method_data.empty:
            continue
        style = style_for_method(method)
        ratio = np.asarray(method_data["ratio"], dtype=float)
        ami_mean = np.asarray(method_data["ami_mean"], dtype=float)
        ami_std = np.nan_to_num(np.asarray(method_data["ami_std"], dtype=float), nan=0.0)
        plt.plot(
            ratio,
            ami_mean,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=6,
            linewidth=2,
            alpha=1,
        )
        plt.fill_between(ratio, ami_mean - ami_std, ami_mean + ami_std, color=style["color"], alpha=0.2)

    plt.xlabel(r"Density Ratio ($n_{\mathrm{low}} / n_{\mathrm{high}}$)", fontsize=12)
    plt.ylabel("AMI Score", fontsize=12)
    plt.legend(loc="best", fontsize=10, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot results from grid-imbalance benchmark")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/benchmark_grid_imbalance_grid_search",
        help="Path to results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to plots/imbalance/<experiment_name>/.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_METHODS_TO_PLOT,
        help="Methods to plot (default: all methods)",
    )
    args = parser.parse_args()

    results_path = project_path(args.results_dir)
    output_dir = resolve_output_dir(args.output_dir, "imbalance", results_path)

    print(f"Loading results from: {results_path}")
    df = load_grid_imbalance_results(results_path)
    if df.empty:
        print("No results found!")
        return

    print(f"Loaded {len(df)} result entries")
    all_methods = sorted(df["method"].unique())
    print(f"Methods: {all_methods}")

    # Filter by selected methods if specified
    if args.methods is not None:
        selected_methods = [m for m in args.methods]  # Preserve user's order
        missing_methods = set(selected_methods) - set(all_methods)
        if missing_methods:
            print(f"Warning: Methods not found in results: {sorted(missing_methods)}")
        available_selected = [m for m in selected_methods if m in all_methods]
        if not available_selected:
            print("Error: No selected methods found in results!")
            return
        df = df[df["method"].isin(available_selected)].copy()
        print(f"Filtered to {len(available_selected)} method(s): {available_selected}")
    grid_sizes = sorted(df[["grid_rows", "grid_cols"]].drop_duplicates().itertuples(index=False, name=None))
    print(f"Grid sizes found: {grid_sizes}")

    for grid_rows, grid_cols in grid_sizes:
        df_grid = df[(df["grid_rows"] == grid_rows) & (df["grid_cols"] == grid_cols)].copy()
        print(f"\n{'=' * 60}")
        print(f"Processing grid size: {grid_rows}x{grid_cols}")
        print(f"{'=' * 60}")
        print(f"Datasets: {len(set(df_grid['seed']))} seeds")
        print(f"Density ratios: {sorted(set(df_grid['ratio']))}")

        output_file = output_dir / f"grid_imbalance_{grid_rows}x{grid_cols}.pdf"
        plot_imbalance_results(pd.DataFrame(df_grid), output_file)

    print(f"\n{'=' * 60}")
    print(f"All plots saved to: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
