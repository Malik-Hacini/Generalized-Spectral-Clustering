"""
Summarize benchmark_uci experiment results.

Compare AMI performance when optimizing for different unsupervised metrics:
- CH (Calinski-Harabasz)
- Graph-CH (with polynomial filter)
"""

import json
import pandas as pd
from pathlib import Path
import sys


def load_optimization_results(results_path: str | Path, optimize_by: str) -> dict:
    """
    Load results for a specific optimization metric.

    Parameters
    ----------
    results_path : str or Path
        Path to the benchmark_uci_grid_search results directory
    optimize_by : str
        Which metric was used for optimization ("ch" or "graph_ch")

    Returns
    -------
    dict
        Dictionary with 'ami_values' DataFrame and 'filter_coeffs' (for graph_ch)
    """
    results_dir = Path(results_path)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find all best_results.json files
    best_result_files = sorted(results_dir.glob("*/*/*_best_results.json"))

    if not best_result_files:
        raise ValueError(f"No best_results.json files found in {results_dir}")

    ami_rows = []
    filter_coeffs_dict = {}

    for best_file in best_result_files:
        method_name = best_file.parent.name
        dataset_name = best_file.parent.parent.name

        with open(best_file, 'r') as f:
            best_results = json.load(f)

        # Extract AMI value when optimizing for the specified metric
        if optimize_by in best_results:
            optimized_data = best_results[optimize_by]

            # Get AMI value
            if "ami" in optimized_data and "mean" in optimized_data["ami"]:
                ami_value = optimized_data["ami"]["mean"]
                ami_rows.append({
                    "dataset": dataset_name,
                    "method": method_name,
                    "ami": ami_value
                })

            # For graph_ch, also extract filter coefficients
            if optimize_by == "graph_ch":
                metric_params = optimized_data.get("metric_params", {})
                if isinstance(metric_params, dict) and "filter_coeffs" in metric_params:
                    key = (dataset_name, method_name)
                    filter_coeffs_dict[key] = metric_params["filter_coeffs"]

    result = {
        'ami_values': pd.DataFrame(ami_rows) if ami_rows else pd.DataFrame(columns=["dataset", "method", "ami"]),
    }

    if optimize_by == "graph_ch":
        result['filter_coeffs'] = filter_coeffs_dict

    return result


def print_ami_comparison(results_path: str | Path):
    """
    Print AMI comparison when optimizing for different unsupervised metrics.

    Parameters
    ----------
    results_path : str or Path
        Path to the benchmark_uci_grid_search results directory
    """
    print("="*80)
    print("AMI COMPARISON: Unsupervised Optimization Strategies")
    print("="*80)
    print()

    # Load results for CH optimization
    ch_results = load_optimization_results(results_path, optimize_by="ch")
    ami_ch_df = ch_results['ami_values']

    # Load results for graph_ch optimization
    graph_ch_results = load_optimization_results(results_path, optimize_by="graph_ch")
    ami_graph_ch_df = graph_ch_results['ami_values']
    filter_coeffs = graph_ch_results.get('filter_coeffs', {})

    # ========== 1. AMI when optimizing for CH ==========
    print("="*80)
    print("1. AMI (Optimized for Calinski-Harabasz)")
    print("="*80)
    print()

    if not ami_ch_df.empty:
        pivot_ch = ami_ch_df.pivot(index="dataset", columns="method", values="ami")
        pivot_ch = pivot_ch.sort_index()
        print(pivot_ch.to_string(float_format=lambda x: f"{x:.3f}"))

        print(f"\n{'-'*80}")
        print("Average AMI by Method (CH-optimized):")
        print(f"{'-'*80}")
        avg_ami_ch = pivot_ch.mean().sort_values(ascending=False)
        for method, value in avg_ami_ch.items():
            print(f"  {method:20s}: {value:.3f}")
    else:
        print("No results found for CH optimization.")

    # ========== 2. AMI when optimizing for Graph-CH ==========
    print("\n\n" + "="*80)
    print("2. AMI (Optimized for Graph-CH)")
    print("="*80)
    print()

    if not ami_graph_ch_df.empty:
        pivot_graph_ch = ami_graph_ch_df.pivot(index="dataset", columns="method", values="ami")
        pivot_graph_ch = pivot_graph_ch.sort_index()
        print(pivot_graph_ch.to_string(float_format=lambda x: f"{x:.3f}"))

        print(f"\n{'-'*80}")
        print("Average AMI by Method (Graph-CH-optimized):")
        print(f"{'-'*80}")
        avg_ami_graph_ch = pivot_graph_ch.mean().sort_values(ascending=False)
        for method, value in avg_ami_graph_ch.items():
            print(f"  {method:20s}: {value:.3f}")

        # Show filter coefficients used
        print(f"\n{'-'*80}")
        print("Graph-CH Filter Coefficients (polynomial filter g(P)):")
        print(f"{'-'*80}")
        if filter_coeffs:
            # Group by unique filter configurations
            unique_filters = {}
            for (dataset, method), coeffs in filter_coeffs.items():
                coeffs_str = str(coeffs)
                if coeffs_str not in unique_filters:
                    unique_filters[coeffs_str] = []
                unique_filters[coeffs_str].append((dataset, method))

            for i, (coeffs_str, instances) in enumerate(unique_filters.items(), 1):
                print(f"\n  Configuration {i}: {coeffs_str}")
                print(f"    Used by {len(instances)} dataset-method combinations")
                if len(unique_filters) == 1:
                    print(f"    (All methods used the same filter)")
        else:
            print("  No filter coefficient information found.")
    else:
        print("No results found for Graph-CH optimization.")

    # ========== 3. Comparison ==========
    if not ami_ch_df.empty and not ami_graph_ch_df.empty:
        print("\n\n" + "="*80)
        print("3. COMPARISON: CH vs Graph-CH Optimization")
        print("="*80)
        print()

        comparison_rows = []
        common_methods = set(ami_ch_df['method'].unique()) & set(ami_graph_ch_df['method'].unique())

        for method in sorted(common_methods):
            ami_ch = ami_ch_df[ami_ch_df['method'] == method]['ami'].mean()
            ami_graph_ch = ami_graph_ch_df[ami_graph_ch_df['method'] == method]['ami'].mean()
            diff = ami_graph_ch - ami_ch
            diff_pct = (diff / ami_ch * 100) if ami_ch > 0 else 0

            comparison_rows.append({
                'Method': method,
                'AMI (CH-opt)': ami_ch,
                'AMI (Graph-CH-opt)': ami_graph_ch,
                'Difference': diff,
                'Diff (%)': diff_pct
            })

        comparison_df = pd.DataFrame(comparison_rows).sort_values('Difference', ascending=False)
        print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        print(f"\n{'-'*80}")
        print("Summary:")
        print(f"{'-'*80}")
        improvements = sum(1 for row in comparison_rows if row['Difference'] > 0)
        degradations = sum(1 for row in comparison_rows if row['Difference'] < 0)
        print(f"  Methods improved by Graph-CH: {improvements}/{len(comparison_rows)}")
        print(f"  Methods degraded by Graph-CH: {degradations}/{len(comparison_rows)}")
        avg_diff = comparison_df['Difference'].mean()
        print(f"  Average difference: {avg_diff:+.3f} ({avg_diff/comparison_df['AMI (CH-opt)'].mean()*100:+.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = "results/benchmark_uci_grid_search"

    print("="*80)
    print("BENCHMARK UCI RESULTS SUMMARY")
    print("="*80)
    print()

    print_ami_comparison(results_path)
