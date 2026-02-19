#!/usr/bin/env python3
"""
KNN Graph Property Analysis for UCI and Network Benchmark Datasets.

Loads UCI point-cloud datasets (builds directed KNN graphs, K = ceil(log N))
and network datasets (pre-built directed adjacency matrices), then generates
publication-quality plots of:
    1. In-degree distributions (one subplot per dataset)
    2. Standard reciprocity and C-L reciprocity (bar chart across datasets)

Separate plots are generated for each dataset family (UCI vs networks).
Results are saved to distribution/results/.

Usage
-----
    source .venv/bin/activate
    python distribution/analysis_knn_properties.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.file_manager import load_dataset
from competitors.neighbors import log_neighbors
from distribution.knn_analysis_utils import analyze_graph, analyze_pointcloud

# =============================================================================
# Configuration
# =============================================================================

UCI_DATASET_NAMES = [
    "breast_tissue", "wine", "control_chart", "glass", "iris",
    "parkinsons", "seeds", "segmentation", "vertebral", "wdbc", "yeast",
]

NETWORK_DATASET_NAMES = [
    "email_eu_core", "polblogs", "wiki_vote",
]

LOAD_PATH = "datasets"
RESULTS_PATH = "distribution/results"

# Plot style (consistent with existing project plots)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'hist': '#1f77b4',
    'mean_line': '#d62728',
    'k_line': '#2ca02c',
    'reciprocity': '#1f77b4',
    'cl_reciprocity': '#ff7f0e',
}


# =============================================================================
# Data Collection
# =============================================================================

def collect_uci_analyses(dataset_names: list[str], load_path: str) -> dict:
    """
    Load each UCI point-cloud dataset and run KNN graph analysis.

    Returns dict mapping dataset_name -> analysis results dict.
    Skips datasets that fail to load with a warning.
    """
    results = {}
    for name in dataset_names:
        try:
            X, y = load_dataset(load_path, name)
        except Exception as e:
            print(f"  Warning: skipping '{name}': {e}")
            continue

        K = log_neighbors(X)
        analysis = analyze_pointcloud(X, y, K)
        results[name] = analysis

        print(
            f"  {name:>16s}: N={analysis['n_samples']:>5d}, "
            f"D={analysis['n_features']:>2d}, K={K}, "
            f"k={analysis['n_clusters']}, "
            f"recip={analysis['reciprocity']:.3f}, "
            f"CL={analysis['cl_reciprocity']:.3f}"
        )

    return results


def collect_network_analyses(dataset_names: list[str], load_path: str) -> dict:
    """
    Load each network dataset and run directed graph analysis.

    Network datasets are pre-built adjacency matrices (no KNN construction).
    Returns dict mapping dataset_name -> analysis results dict.
    Skips datasets that fail to load with a warning.
    """
    results = {}
    for name in dataset_names:
        try:
            A, labels = load_dataset(load_path, name)
        except Exception as e:
            print(f"  Warning: skipping '{name}': {e}")
            continue

        analysis = analyze_graph(A, labels)
        results[name] = analysis

        print(
            f"  {name:>16s}: N={analysis['n_samples']:>5d}, "
            f"nnz={analysis['n_edges']:>6d}, "
            f"k={analysis['n_clusters']}, "
            f"mean_out={analysis['mean_out_degree']:.1f}, "
            f"recip={analysis['reciprocity']:.3f}, "
            f"CL={analysis['cl_reciprocity']:.3f}"
        )

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_in_degree_distributions(
    results: dict,
    save_path: str,
    title: str = "In-Degree Distributions",
    filename_prefix: str = "in_degree_distributions",
) -> None:
    """
    Plot in-degree histograms for all datasets in a grid layout.

    For point-cloud datasets, each subplot shows a K vertical line
    (fixed out-degree). For network datasets, a mean out-degree line
    is shown instead.

    Parameters
    ----------
    results : dict
        Mapping of dataset_name -> analysis results dict.
    save_path : str
        Directory to save PDF and PNG files.
    title : str
        Overall figure suptitle.
    filename_prefix : str
        Prefix for saved filenames (produces {prefix}.pdf and {prefix}.png).
    """
    n_datasets = len(results)
    n_cols = 4
    n_rows = int(np.ceil(n_datasets / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        in_deg = data['in_degrees']
        is_graph = data['dataset_type'] == 'graph'

        if is_graph:
            # Network: wider degree range, use auto-binning
            ax.hist(
                in_deg, bins='auto', color=COLORS['hist'],
                alpha=0.7, edgecolor='white', linewidth=0.5, density=True,
            )
            mean_out = data['mean_out_degree']
            ax.axvline(mean_out, color=COLORS['mean_line'], linestyle='--',
                       linewidth=1.5, label=f'mean out={mean_out:.1f}')
            subtitle = f"(N={data['n_samples']}, nnz={data['n_edges']})"
        else:
            # Point cloud: integer-aligned unit-width bins, K line only
            K = data['n_neighbors']
            bins = np.arange(in_deg.min() - 0.5, in_deg.max() + 1.5, 1)
            ax.hist(
                in_deg, bins=bins, color=COLORS['hist'],
                alpha=0.7, edgecolor='white', linewidth=0.5, density=True,
            )
            ax.axvline(K, color=COLORS['k_line'], linestyle=':',
                       linewidth=1.5, label=f'K={K}')
            subtitle = f"(N={data['n_samples']}, K={K})"

        ax.set_title(f"{name}\n{subtitle}", fontsize=10)
        ax.set_xlabel('In-degree')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7, loc='upper right')

    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()

    fig.savefig(f"{save_path}/{filename_prefix}.pdf", bbox_inches='tight')
    fig.savefig(f"{save_path}/{filename_prefix}.png", bbox_inches='tight', dpi=300)
    print(f"\n  Saved: {filename_prefix}.pdf/png")
    plt.close(fig)


def plot_reciprocity_comparison(
    results: dict,
    save_path: str,
    title: str = "Standard vs C-L Reciprocity",
    filename_prefix: str = "cl_reciprocity",
) -> None:
    """
    Bar chart comparing standard reciprocity and C-L reciprocity across datasets.

    Parameters
    ----------
    results : dict
        Mapping of dataset_name -> analysis results dict.
    save_path : str
        Directory to save PDF and PNG files.
    title : str
        Chart title.
    filename_prefix : str
        Prefix for saved filenames (produces {prefix}.pdf and {prefix}.png).
    """
    names = list(results.keys())
    recip = [results[n]['reciprocity'] for n in names]
    cl_recip = [results[n]['cl_reciprocity'] for n in names]

    x = np.arange(len(names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))

    bars_recip = ax.bar(
        x - bar_width / 2, recip, bar_width,
        color=COLORS['reciprocity'], alpha=0.8,
        label='Standard Reciprocity', edgecolor='white', linewidth=0.5,
    )
    bars_cl = ax.bar(
        x + bar_width / 2, cl_recip, bar_width,
        color=COLORS['cl_reciprocity'], alpha=0.8,
        label='C-L Reciprocity', edgecolor='white', linewidth=0.5,
    )

    # Value annotations
    for bar in bars_recip:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7,
        )
    for bar in bars_cl:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Reciprocity')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_ylim(0, min(1.15, max(max(recip), max(cl_recip)) + 0.15))

    plt.tight_layout()
    fig.savefig(f"{save_path}/{filename_prefix}.pdf", bbox_inches='tight')
    fig.savefig(f"{save_path}/{filename_prefix}.png", bbox_inches='tight', dpi=300)
    print(f"  Saved: {filename_prefix}.pdf/png")
    plt.close(fig)


def save_summary_table(
    results: dict, save_path: str, filename: str = "summary.csv",
) -> pd.DataFrame:
    """
    Save a CSV summary of all dataset analysis results.

    Handles both point-cloud and network dataset types, using appropriate
    columns for each (D/K for point clouds, nnz/mean_out_degree for networks).

    Parameters
    ----------
    results : dict
        Mapping of dataset_name -> analysis results dict.
    save_path : str
        Directory to save the CSV file.
    filename : str
        Name of the CSV file.

    Returns
    -------
    pd.DataFrame
        Summary table.
    """
    rows = []
    for name, data in results.items():
        in_deg = data['in_degrees']
        row = {
            'dataset': name,
            'type': data['dataset_type'],
            'N': data['n_samples'],
            'k_clusters': data['n_clusters'],
            'in_degree_mean': round(float(in_deg.mean()), 2),
            'in_degree_std': round(float(in_deg.std()), 2),
            'in_degree_min': int(in_deg.min()),
            'in_degree_max': int(in_deg.max()),
            'reciprocity': round(data['reciprocity'], 4),
            'cl_reciprocity': round(data['cl_reciprocity'], 4),
        }

        if data['dataset_type'] == 'pointcloud':
            row['D'] = data['n_features']
            row['K'] = data['n_neighbors']
        else:
            row['nnz'] = data['n_edges']
            row['mean_out_degree'] = round(data['mean_out_degree'], 2)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{save_path}/{filename}", index=False)
    print(f"  Saved: {filename}")
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("KNN GRAPH PROPERTY ANALYSIS")
    print("=" * 60)

    os.makedirs(RESULTS_PATH, exist_ok=True)

    # ----- UCI point-cloud datasets -----
    print(f"\nUCI Datasets: {UCI_DATASET_NAMES}")
    print(f"Neighbor rule: K = ceil(log N)")
    print(f"Output: {RESULTS_PATH}/\n")

    print("--- Analyzing UCI datasets ---")
    uci_results = collect_uci_analyses(UCI_DATASET_NAMES, LOAD_PATH)

    if uci_results:
        print("\n--- Generating UCI plots ---")
        plot_in_degree_distributions(
            uci_results, RESULTS_PATH,
            title="In-Degree Distributions of Directed KNN Graphs",
            filename_prefix="in_degree_distributions",
        )
        plot_reciprocity_comparison(
            uci_results, RESULTS_PATH,
            title="Standard vs C-L Reciprocity Across UCI Datasets",
            filename_prefix="cl_reciprocity",
        )
        uci_df = save_summary_table(uci_results, RESULTS_PATH, "summary.csv")
        print(f"\n{uci_df.to_string(index=False)}")
    else:
        print("No UCI datasets loaded successfully.")

    # ----- Network datasets -----
    print(f"\n{'=' * 60}")
    print(f"Network Datasets: {NETWORK_DATASET_NAMES}")
    print(f"Output: {RESULTS_PATH}/\n")

    print("--- Analyzing network datasets ---")
    net_results = collect_network_analyses(NETWORK_DATASET_NAMES, LOAD_PATH)

    if net_results:
        print("\n--- Generating network plots ---")
        plot_in_degree_distributions(
            net_results, RESULTS_PATH,
            title="In-Degree Distributions of Directed Network Graphs",
            filename_prefix="networks_in_degree_distributions",
        )
        plot_reciprocity_comparison(
            net_results, RESULTS_PATH,
            title="Standard vs C-L Reciprocity Across Network Datasets",
            filename_prefix="networks_cl_reciprocity",
        )
        net_df = save_summary_table(
            net_results, RESULTS_PATH, "networks_summary.csv",
        )
        print(f"\n{net_df.to_string(index=False)}")
    else:
        print("No network datasets loaded successfully.")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll results saved to: {RESULTS_PATH}/")


if __name__ == '__main__':
    main()
