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
        DataFrame with columns: method, n_high, n_low, ratio, seed, ami
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

        # Parse dataset name: grid_2x2_high300_low20_seed0
        match = re.match(r'grid_\d+x\d+_high(\d+)_low(\d+)_seed(\d+)', dataset_name)
        if not match:
            print(f"Warning: Could not parse dataset name: {dataset_name}")
            continue

        n_high = int(match.group(1))
        n_low = int(match.group(2))
        seed = int(match.group(3))
        ratio = n_low / n_high

        with open(best_file, 'r') as f:
            best_results = json.load(f)

        # Get AMI score (optimized for AMI if available, otherwise use first available)
        ami_value = None
        if "ami" in best_results:
            optimized_data = best_results["ami"]
            if "ami" in optimized_data and "mean" in optimized_data["ami"]:
                ami_value = optimized_data["ami"]["mean"]

        if ami_value is not None:
            rows.append({
                "method": method_name,
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

    plt.xlabel('Density Ratio ($n_{\mathrm{low}} / n_{\mathrm{high}}$)', fontsize=12)
    plt.ylabel('AMI Score', fontsize=12)
    plt.title('Clustering Performance vs Cluster Density Imbalance', fontsize=14, fontweight='bold')
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
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for plot (e.g., plot.png)')
    parser.add_argument('--show-stats', action='store_true',
                       help='Print summary statistics')

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_path}")
    df = load_grid_imbalance_results(args.results_path)

    if df.empty:
        print("No results found!")
        return

    print(f"Loaded {len(df)} result entries")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Density ratios: {sorted(df['ratio'].unique())}")
    print(f"Seeds per configuration: {df.groupby(['method', 'ratio'])['seed'].nunique().iloc[0]}")
    print()

    # Print statistics if requested
    if args.show_stats:
        print_summary_statistics(df)
        print()

    # Create plot
    plot_imbalance_results(df, args.output)


if __name__ == "__main__":
    main()
