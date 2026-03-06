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

    # Load results
    print("Loading results...")
    df = summarize_uci_results(results_path)

    print(f"Loaded {len(df)} method-dataset combinations")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Methods: {sorted(df['method'].unique())}")

    # Print AMI summary
    print_summary_table(df, metric="ami", show_std=False)

    # Print top methods per dataset
    print_best_methods_per_dataset(df, metric="ami", top_n=3)

    # Print optimized parameters
    print_optimized_parameters(results_path)

    # Print CH summary
    print_summary_table(df, metric="ch", show_std=False)
