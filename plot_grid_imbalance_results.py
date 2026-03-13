"""
Plot AMI scores vs imbalance ratio for grid-imbalance benchmark.

Creates a line plot showing mean AMI ± std for each clustering method
as a function of the n_low/n_high density ratio.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_grid_imbalance_results(results_path: str | Path):
    """
    Load results from grid-imbalance benchmark.

    Parameters
    ----------
    results_path : str or Path
        Path to the benchmark_grid_imbalance results directory

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: method, grid_rows, grid_cols, n_high, n_low, ratio, seed, ami
    """
    results_dir = Path(results_path)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find all best_results.json files
    best_result_files = sorted(results_dir.glob("*/*/*_best_results.json"))

    if not best_result_files:
        raise ValueError(f"No best_results.json files found in {results_dir}")

    rows = []

    for best_file in best_result_files:
        method_name = best_file.parent.name
        dataset_name = best_file.parent.parent.name

        # Parse dataset names from both legacy and tuple-based formats:
        # - grid_2x2_high300_low20_seed0
        # - grid_(2, 1)x(2, 1)_high300_low20_seed0
        square_match = re.match(
            r'grid_(\d+)x(\d+)_high(\d+)_low(\d+)_seed(\d+)$',
            dataset_name,
        )
        tuple_match = re.match(
            r'grid_\((\d+),\s*(\d+)\)x\((\d+),\s*(\d+)\)_high(\d+)_low(\d+)_seed(\d+)$',
            dataset_name,
        )

        if square_match:
            grid_rows = int(square_match.group(1))
            grid_cols = int(square_match.group(2))
            n_high = int(square_match.group(3))
            n_low = int(square_match.group(4))
            seed = int(square_match.group(5))
        elif tuple_match:
            left_rows = int(tuple_match.group(1))
            left_cols = int(tuple_match.group(2))
            right_rows = int(tuple_match.group(3))
            right_cols = int(tuple_match.group(4))

            # Dataset names duplicate the same shape on both sides; keep left side if they differ.
            if (left_rows, left_cols) != (right_rows, right_cols):
                print(
                    "Warning: Inconsistent grid shape in dataset name "
                    f"{dataset_name}; using ({left_rows}, {left_cols})."
                )

            grid_rows, grid_cols = left_rows, left_cols
            n_high = int(tuple_match.group(5))
            n_low = int(tuple_match.group(6))
            seed = int(tuple_match.group(7))
        else:
            print(f"Warning: Could not parse dataset name: {dataset_name}")
            continue

        ratio = n_low / n_high

        with open(best_file, 'r') as f:
            best_results = json.load(f)

        # Get AMI score (optimized by CH if available, otherwise use AMI)
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
            rows.append({
                "method": method_name,
                "grid_rows": grid_rows,
                "grid_cols": grid_cols,
                "n_high": n_high,
                "n_low": n_low,
                "ratio": ratio,
                "seed": seed,
                "ami": ami_value
            })

    return pd.DataFrame(rows)


def plot_imbalance_results(df: pd.DataFrame, output_file: str = None):
    """
    Create line plot of AMI vs density ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame from load_grid_imbalance_results
    output_file : str, optional
        Path to save the plot
    """
    # Aggregate by method and ratio
    summary = df.groupby(['method', 'ratio']).agg({
        'ami': ['mean', 'std', 'count']
    }).reset_index()

    summary.columns = ['method', 'ratio', 'ami_mean', 'ami_std', 'n_seeds']

    # Sort by ratio for proper line plotting
    summary = summary.sort_values('ratio')

    # Define method order and styling
    method_order = ['SC-N', 'DSC+', 'GSC-N']
    method_styles = {
        'SC-N': {'color': '#FF6347', 'linestyle': '--', 'marker': 'o', 'label': 'SC-N'},
        'DSC+': {'color': "#27A727", 'linestyle': '-.', 'marker': '^', 'label': 'DSC+'},
        'GSC-N': {'color': '#072AC8', 'linestyle': '-', 'marker': 's', 'label': 'GSC-N'},
    }

    # Create plot
    # plt.figure(figsize=(10, 6))
    plt.figure()

    for method in method_order:
        method_data = summary[summary['method'] == method]

        if method_data.empty:
            continue

        style = method_styles.get(method, {})

        ratio = method_data['ratio'].values
        ami_mean = method_data['ami_mean'].values
        ami_std = method_data['ami_std'].values

        # Plot mean line
        plt.plot(
            ratio,
            ami_mean,
            label=style.get('label', method),
            color=style.get('color', None),
            linestyle=style.get('linestyle', '-'),
            marker=style.get('marker', 'o'),
            markersize=6,
            linewidth=2,
            alpha=1
        )

        # Fill between mean ± std
        plt.fill_between(
            ratio,
            ami_mean - ami_std,
            ami_mean + ami_std,
            color=style.get('color', None),
            alpha=0.2
        )

    plt.xlabel(r'Density Ratio ($n_{\mathrm{low}} / n_{\mathrm{high}}$)', fontsize=12)
    plt.ylabel('AMI Score', fontsize=12)
    plt.legend(loc='best', fontsize=10, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics of the results."""
    print("=" * 80)
    print("GRID-IMBALANCE BENCHMARK SUMMARY")
    print("=" * 80)
    print()

    # Overall statistics by method
    overall = df.groupby('method').agg({
        'ami': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)

    print("Overall AMI Statistics by Method:")
    print(overall)
    print()

    # Statistics by ratio
    by_ratio = df.groupby(['ratio', 'method'])['ami'].agg(['mean', 'std']).unstack('method').round(4)

    print("\nAMI by Density Ratio:")
    print(by_ratio)
    print()

    # Find best method for each ratio
    best_by_ratio = df.groupby(['ratio', 'method'])['ami'].mean().unstack('method')
    print("\nBest Method by Density Ratio:")
    for ratio in sorted(best_by_ratio.index):
        best_method = best_by_ratio.loc[ratio].idxmax()
        best_score = best_by_ratio.loc[ratio].max()
        print(f"  Ratio {ratio:.3f}: {best_method} (AMI = {best_score:.4f})")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot results from grid-imbalance benchmark'
    )
    parser.add_argument('--results-path', type=str,
                       default='results/benchmark_grid_imbalance_grid_search',
                       help='Path to results directory')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for plots (default: figures/)')
    parser.add_argument('--show-stats', action='store_true',
                       help='Print summary statistics')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from: {args.results_path}")
    df = load_grid_imbalance_results(args.results_path)

    if df.empty:
        print("No results found!")
        return

    print(f"Loaded {len(df)} result entries")
    print(f"Methods: {sorted(df['method'].unique())}")

    # Get all unique grid sizes as (rows, cols)
    grid_sizes = sorted(
        df[['grid_rows', 'grid_cols']].drop_duplicates().itertuples(index=False, name=None)
    )
    print(f"Grid sizes found: {grid_sizes}")
    print()

    # Process each grid size separately
    for grid_rows, grid_cols in grid_sizes:
        df_grid = df[(df['grid_rows'] == grid_rows) & (df['grid_cols'] == grid_cols)]

        print(f"\n{'='*60}")
        print(f"Processing grid size: {grid_rows}x{grid_cols}")
        print(f"{'='*60}")
        print(f"Datasets: {len(df_grid['seed'].unique())} seeds")
        print(f"Density ratios: {sorted(df_grid['ratio'].unique())}")

        # Print statistics if requested
        if args.show_stats:
            print()
            print_summary_statistics(df_grid)
            print()

        # Create plot for this grid size
        output_file = output_dir / f"grid_imbalance_{grid_rows}x{grid_cols}.pdf"
        plot_imbalance_results(df_grid, str(output_file))
        print(f"Saved plot to: {output_file}")

    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
