"""
Summarize benchmark_uci experiment results.

This script loads all dataset summaries from a benchmark_uci_grid_search directory
and displays them in a comprehensive table.
"""

import json
import pandas as pd
from pathlib import Path
import sys

def summarize_uci_results(results_path: str | Path) -> pd.DataFrame:
    """
    Load and summarize benchmark_uci experiment results.

    Parameters
    ----------
    results_path : str or Path
        Path to the benchmark_uci_grid_search results directory
        (e.g., "results/benchmark_uci_grid_search")

    Returns
    -------
    pd.DataFrame
        Combined summary table with all datasets and methods
    """
    results_dir = Path(results_path)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find all summary CSV files
    summary_files = sorted(results_dir.glob("*/*_summary.csv"))

    if not summary_files:
        raise ValueError(f"No summary files found in {results_dir}")

    # Load and combine all summaries
    summaries = []
    for summary_file in summary_files:
        dataset_name = summary_file.parent.name
        df = pd.read_csv(summary_file)
        df.insert(0, "dataset", dataset_name)
        summaries.append(df)

    # Combine all summaries
    combined_df = pd.concat(summaries, ignore_index=True)

    return combined_df


def print_summary_table(df: pd.DataFrame, metric: str = "ami", show_std: bool = False):
    """
    Print a formatted summary table for a specific metric.

    Parameters
    ----------
    df : pd.DataFrame
        Combined summary dataframe from summarize_uci_results()
    metric : str
        Metric to display ("ami", "ch", etc.)
    show_std : bool
        Whether to show standard deviation columns
    """
    # Determine which columns to show
    best_mean_col = f"{metric}_best_mean"
    best_std_col = f"{metric}_best_std"

    if best_mean_col not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results. Available columns: {df.columns.tolist()}")

    # Create pivot table
    if show_std:
        pivot_mean = df.pivot(index="dataset", columns="method", values=best_mean_col)
        pivot_std = df.pivot(index="dataset", columns="method", values=best_std_col)

        print(f"\n{'='*80}")
        print(f"Best {metric.upper()} (mean ± std) across all datasets")
        print(f"{'='*80}\n")

        # Combine mean and std for display
        for dataset in pivot_mean.index:
            print(f"\n{dataset}:")
            for method in pivot_mean.columns:
                mean_val = pivot_mean.loc[dataset, method]
                std_val = pivot_std.loc[dataset, method]
                print(f"  {method:20s}: {mean_val:.3f} ± {std_val:.3f}")
    else:
        pivot = df.pivot(index="dataset", columns="method", values=best_mean_col)

        print(f"\n{'='*80}")
        print(f"Best {metric.upper()} (mean) across all datasets")
        print(f"{'='*80}\n")
        print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Summary Statistics - {metric.upper()}")
    print(f"{'='*80}\n")

    method_summary = df.groupby("method")[best_mean_col].agg(['mean', 'std', 'min', 'max'])
    method_summary.columns = ['avg_across_datasets', 'std_across_datasets', 'min', 'max']
    print(method_summary.to_string(float_format=lambda x: f"{x:.3f}"))


def print_best_methods_per_dataset(df: pd.DataFrame, metric: str = "ami", top_n: int = 3):
    """
    Print the top N methods for each dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Combined summary dataframe from summarize_uci_results()
    metric : str
        Metric to rank by ("ami", "ch", etc.)
    top_n : int
        Number of top methods to show per dataset
    """
    best_mean_col = f"{metric}_best_mean"

    print(f"\n{'='*80}")
    print(f"Top {top_n} Methods per Dataset (ranked by {metric.upper()})")
    print(f"{'='*80}\n")

    for dataset in sorted(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset].copy()
        dataset_df = dataset_df.sort_values(best_mean_col, ascending=False).head(top_n)

        print(f"{dataset}:")
        for idx, row in dataset_df.iterrows():
            print(f"  {row['method']:20s}: {row[best_mean_col]:.3f}")
        print()


def load_results_from_json(results_path: str | Path, optimize_by: str = "ami") -> dict:
    """
    Load experiment results from best_results.json files.

    Parameters
    ----------
    results_path : str or Path
        Path to the benchmark_uci_grid_search results directory
    optimize_by : str
        Which metric was used for optimization ("ami" or "ch")

    Returns
    -------
    dict
        Dictionary with keys: 'ami_values', 'ch_values', 'graph_ch_values', 'parameters'
        Each containing DataFrames with dataset, method, and values
    """
    results_dir = Path(results_path)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find all best_results.json files
    best_result_files = sorted(results_dir.glob("*/*/*_best_results.json"))

    if not best_result_files:
        raise ValueError(f"No best_results.json files found in {results_dir}")

    ami_rows = []
    ch_rows = []
    graph_ch_rows = []
    param_rows = []

    for best_file in best_result_files:
        method_name = best_file.parent.name
        dataset_name = best_file.parent.parent.name

        with open(best_file, 'r') as f:
            best_results = json.load(f)

        # Extract values based on optimization target
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

            # Get CH value
            if "ch" in optimized_data and "mean" in optimized_data["ch"]:
                ch_value = optimized_data["ch"]["mean"]
                ch_rows.append({
                    "dataset": dataset_name,
                    "method": method_name,
                    "ch": ch_value
                })

            # Get graph_ch value
            if "graph_ch" in optimized_data and "mean" in optimized_data["graph_ch"]:
                graph_ch_value = optimized_data["graph_ch"]["mean"]
                graph_ch_rows.append({
                    "dataset": dataset_name,
                    "method": method_name,
                    "graph_ch": graph_ch_value
                })

            # Extract parameters
            params = {}
            for key, value in optimized_data.items():
                if key in ["ami", "ch", "graph_ch", "predicted_labels"]:
                    continue

                # Handle measure parameters (for GSC methods)
                if key == "measure" and isinstance(value, list) and len(value) >= 2:
                    if isinstance(value[1], dict):
                        for param_name, param_value in value[1].items():
                            params[f"measure_{param_name}"] = param_value

                # Handle other parameters (gamma, tau, etc.)
                elif not isinstance(value, (list, dict)):
                    params[key] = value

            if params:  # Only add if there are optimized parameters
                param_rows.append({
                    "dataset": dataset_name,
                    "method": method_name,
                    "optimize_by": optimize_by,
                    **params
                })

    return {
        'ami_values': pd.DataFrame(ami_rows) if ami_rows else pd.DataFrame(columns=["dataset", "method", "ami"]),
        'ch_values': pd.DataFrame(ch_rows) if ch_rows else pd.DataFrame(columns=["dataset", "method", "ch"]),
        'graph_ch_values': pd.DataFrame(graph_ch_rows) if graph_ch_rows else pd.DataFrame(columns=["dataset", "method", "graph_ch"]),
        'parameters': pd.DataFrame(param_rows) if param_rows else pd.DataFrame(columns=["dataset", "method", "optimize_by"])
    }


def print_results_comparison(results_path: str | Path):
    """
    Print comprehensive comparison of results optimized by different metrics.

    Shows three scenarios:
    1. Best AMI (supervised) - AMI when optimizing for AMI
    2. Best CH - CH when optimizing for CH
    3. Best AMI (unsupervised) - AMI when optimizing for CH

    Parameters
    ----------
    results_path : str or Path
        Path to the benchmark_uci_grid_search results directory
    """
    # Load results optimized by AMI
    ami_opt = load_results_from_json(results_path, optimize_by="ami")

    # Load results optimized by CH
    ch_opt = load_results_from_json(results_path, optimize_by="ch")

    # ========== 1. BEST AMI (SUPERVISED) ==========
    print("\n" + "="*80)
    print("1. BEST AMI (SUPERVISED) - Optimizing for AMI")
    print("="*80 + "\n")

    ami_supervised_df = ami_opt['ami_values']
    if not ami_supervised_df.empty:
        pivot_ami_sup = ami_supervised_df.pivot(index="dataset", columns="method", values="ami")
        pivot_ami_sup = pivot_ami_sup.sort_index()
        print(pivot_ami_sup.to_string(float_format=lambda x: f"{x:.3f}"))

        # Summary statistics
        print(f"\n{'-'*80}")
        print("Average AMI by Method (supervised):")
        print(f"{'-'*80}")
        avg_ami = pivot_ami_sup.mean().sort_values(ascending=False)
        for method, value in avg_ami.items():
            print(f"  {method:20s}: {value:.3f}")

    # Parameters for AMI optimization
    print(f"\n{'-'*80}")
    print("Selected Parameters (AMI-optimized):")
    print(f"{'-'*80}")
    _print_parameters_compact(ami_opt['parameters'])

    # ========== 2. BEST CH ==========
    print("\n\n" + "="*80)
    print("2. BEST CH - Optimizing for Calinski-Harabasz")
    print("="*80 + "\n")

    ch_df = ch_opt['ch_values']
    if not ch_df.empty:
        pivot_ch = ch_df.pivot(index="dataset", columns="method", values="ch")
        pivot_ch = pivot_ch.sort_index()
        print(pivot_ch.to_string(float_format=lambda x: f"{x:.1f}"))

        # Summary statistics
        print(f"\n{'-'*80}")
        print("Average CH by Method:")
        print(f"{'-'*80}")
        avg_ch = pivot_ch.mean().sort_values(ascending=False)
        for method, value in avg_ch.items():
            print(f"  {method:20s}: {value:.1f}")

    # Parameters for CH optimization
    print(f"\n{'-'*80}")
    print("Selected Parameters (CH-optimized):")
    print(f"{'-'*80}")
    _print_parameters_compact(ch_opt['parameters'])

    # ========== 3. BEST AMI (UNSUPERVISED) ==========
    print("\n\n" + "="*80)
    print("3. BEST AMI (UNSUPERVISED) - AMI when optimizing for CH")
    print("="*80 + "\n")

    ami_unsupervised_df = ch_opt['ami_values']
    ch_unsupervised_df = ch_opt['ch_values']
    graph_ch_unsupervised_df = ch_opt['graph_ch_values']

    if not ami_unsupervised_df.empty:
        pivot_ami_unsup = ami_unsupervised_df.pivot(index="dataset", columns="method", values="ami")
        pivot_ami_unsup = pivot_ami_unsup.sort_index()
        print("AMI (unsupervised):")
        print(pivot_ami_unsup.to_string(float_format=lambda x: f"{x:.3f}"))

        # Summary statistics
        print(f"\n{'-'*80}")
        print("Average AMI by Method (unsupervised):")
        print(f"{'-'*80}")
        avg_ami_unsup = pivot_ami_unsup.mean().sort_values(ascending=False)
        for method, value in avg_ami_unsup.items():
            print(f"  {method:20s}: {value:.3f}")

    # Show graph-CH values for the unsupervised scenario
    if not graph_ch_unsupervised_df.empty:
        print(f"\n{'-'*80}")
        print("Graph-CH (CH-optimized parameters):")
        print(f"{'-'*80}\n")
        pivot_graph_ch_unsup = graph_ch_unsupervised_df.pivot(index="dataset", columns="method", values="graph_ch")
        pivot_graph_ch_unsup = pivot_graph_ch_unsup.sort_index()
        print(pivot_graph_ch_unsup.to_string(float_format=lambda x: f"{x:.3f}"))

        # Summary statistics for graph-CH
        print(f"\n{'-'*80}")
        print("Average Graph-CH by Method (unsupervised):")
        print(f"{'-'*80}")
        avg_graph_ch_unsup = pivot_graph_ch_unsup.mean().sort_values(ascending=False)
        for method, value in avg_graph_ch_unsup.items():
            print(f"  {method:20s}: {value:.3f}")


    # Also show CH values for the unsupervised scenario
    if not ch_unsupervised_df.empty:
        print(f"\n{'-'*80}")
        print("CH (optimized, same parameters as above):")
        print(f"{'-'*80}\n")
        pivot_ch_unsup = ch_unsupervised_df.pivot(index="dataset", columns="method", values="ch")
        pivot_ch_unsup = pivot_ch_unsup.sort_index()
        print(pivot_ch_unsup.to_string(float_format=lambda x: f"{x:.1f}"))

        # Summary statistics for CH
        print(f"\n{'-'*80}")
        print("Average CH by Method (unsupervised):")
        print(f"{'-'*80}")
        avg_ch_unsup = pivot_ch_unsup.mean().sort_values(ascending=False)
        for method, value in avg_ch_unsup.items():
            print(f"  {method:20s}: {value:.1f}")

    # ========== COMPARISON ==========
    print("\n\n" + "="*80)
    print("COMPARISON: Supervised vs Unsupervised AMI")
    print("="*80 + "\n")

    if not ami_supervised_df.empty and not ami_unsupervised_df.empty:
        comparison_rows = []
        for method in sorted(set(ami_supervised_df['method'].unique()) & set(ami_unsupervised_df['method'].unique())):
            ami_sup = ami_supervised_df[ami_supervised_df['method'] == method]['ami'].mean()
            ami_unsup = ami_unsupervised_df[ami_unsupervised_df['method'] == method]['ami'].mean()
            gap = ami_sup - ami_unsup
            gap_pct = (gap / ami_sup * 100) if ami_sup > 0 else 0

            comparison_rows.append({
                'Method': method,
                'AMI (supervised)': ami_sup,
                'AMI (unsupervised)': ami_unsup,
                'Gap': gap,
                'Gap (%)': gap_pct
            })

        comparison_df = pd.DataFrame(comparison_rows).sort_values('Gap', ascending=False)
        print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _print_parameters_compact(params_df: pd.DataFrame):
    """Helper function to print parameters in a compact format."""
    if params_df.empty:
        print("  No optimized parameters.")
        return

    # Group by method
    methods_with_params = params_df["method"].unique()

    for method in sorted(methods_with_params):
        method_df = params_df[params_df["method"] == method]

        # Get parameter columns (excluding dataset, method, optimize_by)
        param_cols = [col for col in method_df.columns
                     if col not in ["dataset", "method", "optimize_by"]]

        if not param_cols:
            continue

        print(f"\n  {method}:")

        # Show statistics for each parameter
        for param in param_cols:
            values = method_df[param].dropna()
            if len(values) > 0:
                if values.dtype in ['float64', 'int64']:
                    print(f"    {param:18s}: mean={values.mean():6.3f}, "
                          f"std={values.std():6.3f}, "
                          f"range=[{values.min():.3f}, {values.max():.3f}]")
                else:
                    unique_vals = values.unique()
                    if len(unique_vals) <= 5:
                        print(f"    {param:18s}: {unique_vals}")
                    else:
                        print(f"    {param:18s}: {len(unique_vals)} unique values")


def load_best_parameters(results_path: str | Path) -> pd.DataFrame:
    """
    Load best parameters for each method-dataset combination.

    Parameters
    ----------
    results_path : str or Path
        Path to the benchmark_uci_grid_search results directory

    Returns
    -------
    pd.DataFrame
        Table with dataset, method, and optimized parameters
    """
    results_dir = Path(results_path)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find all best_results.json files
    best_result_files = sorted(results_dir.glob("*/*/*_best_results.json"))

    if not best_result_files:
        raise ValueError(f"No best_results.json files found in {results_dir}")

    param_rows = []

    for best_file in best_result_files:
        method_name = best_file.parent.name
        dataset_name = best_file.parent.parent.name

        with open(best_file, 'r') as f:
            best_results = json.load(f)

        # Extract parameters from the best AMI result
        if "ami" in best_results:
            ami_best = best_results["ami"]
            params = {}

            # Extract common optimized parameters
            for key, value in ami_best.items():
                if key in ["ami", "ch", "predicted_labels"]:
                    continue

                # Handle measure parameters (for GSC methods)
                if key == "measure" and isinstance(value, list) and len(value) >= 2:
                    if isinstance(value[1], dict):
                        for param_name, param_value in value[1].items():
                            params[f"measure_{param_name}"] = param_value

                # Handle other parameters (gamma, tau, etc.)
                elif not isinstance(value, (list, dict)):
                    params[key] = value

            if params:  # Only add if there are optimized parameters
                param_rows.append({
                    "dataset": dataset_name,
                    "method": method_name,
                    **params
                })

    if not param_rows:
        return pd.DataFrame(columns=["dataset", "method"])

    return pd.DataFrame(param_rows)


def print_optimized_parameters(results_path: str | Path, methods: list[str] | None = None):
    """
    Print optimized parameters for methods that have grid search.

    Parameters
    ----------
    results_path : str or Path
        Path to the benchmark_uci_grid_search results directory
    methods : list of str, optional
        List of methods to show. If None, shows all methods with parameters.
    """
    params_df = load_best_parameters(results_path)

    if params_df.empty:
        print("No optimized parameters found.")
        return

    if methods is not None:
        params_df = params_df[params_df["method"].isin(methods)]

    # Group by method to see which parameters each method optimizes
    methods_with_params = params_df["method"].unique()

    print(f"\n{'='*80}")
    print("Optimized Parameters by Method")
    print(f"{'='*80}\n")

    for method in sorted(methods_with_params):
        method_df = params_df[params_df["method"] == method]

        # Get parameter columns (excluding dataset and method)
        param_cols = [col for col in method_df.columns if col not in ["dataset", "method"]]

        if not param_cols:
            continue

        print(f"\n{method}:")
        print(f"{'-' * len(method)}")

        # Show parameter values for each dataset
        for _, row in method_df.iterrows():
            param_str = ", ".join([f"{col}={row[col]:.3f}" if isinstance(row[col], float)
                                   else f"{col}={row[col]}"
                                   for col in param_cols])
            print(f"  {row['dataset']:20s}: {param_str}")

    # Print parameter summary statistics
    print(f"\n\n{'='*80}")
    print("Parameter Statistics")
    print(f"{'='*80}\n")

    for method in sorted(methods_with_params):
        method_df = params_df[params_df["method"] == method]
        param_cols = [col for col in method_df.columns if col not in ["dataset", "method"]]

        if not param_cols:
            continue

        print(f"\n{method}:")
        for param in param_cols:
            if param in method_df.columns:
                values = method_df[param].dropna()
                if len(values) > 0:
                    if values.dtype in ['float64', 'int64']:
                        print(f"  {param:20s}: mean={values.mean():.3f}, std={values.std():.3f}, "
                              f"min={values.min():.3f}, max={values.max():.3f}")
                    else:
                        unique_vals = values.unique()
                        print(f"  {param:20s}: unique values = {unique_vals}")



if __name__ == "__main__":
    # Example usage

    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = "results/benchmark_uci_grid_search"

    print("="*80)
    print("BENCHMARK UCI RESULTS SUMMARY")
    print("="*80)

    # Print comprehensive comparison
    print_results_comparison(results_path)
