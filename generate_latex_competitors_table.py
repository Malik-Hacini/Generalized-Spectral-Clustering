"""
Generate LaTeX tables comparing GSC methods with competitors.

Shows AMI scores when parameters are optimized for CH, with the optimized
parameters displayed for methods with hyperparameters.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def load_results_with_params(results_path: str | Path, optimize_by: str = "ch"):
    """
    Load results optimized by a specific metric, including parameters.

    Parameters
    ----------
    results_path : str or Path
        Path to the benchmark_uci_grid_search results directory
    optimize_by : str
        Which metric was used for optimization ("ch", "graph_ch", "ami")

    Returns
    -------
    dict
        Dictionary with DataFrames for ami_values and parameters
    """
    results_dir = Path(results_path)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    best_result_files = sorted(results_dir.glob("*/*/*_best_results.json"))

    if not best_result_files:
        raise ValueError(f"No best_results.json files found in {results_dir}")

    ami_rows = []
    param_rows = []

    for best_file in best_result_files:
        method_name = best_file.parent.name
        dataset_name = best_file.parent.parent.name

        with open(best_file, 'r') as f:
            best_results = json.load(f)

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

            # Extract parameters
            params = {"dataset": dataset_name, "method": method_name}

            # Handle measure parameters for GSC methods
            if "measure" in optimized_data and isinstance(optimized_data["measure"], list):
                if len(optimized_data["measure"]) >= 2 and isinstance(optimized_data["measure"][1], dict):
                    measure_params = optimized_data["measure"][1]
                    for key, value in measure_params.items():
                        params[key] = value

            # Handle other parameters (gamma for DSC+, tau for DI-SIM)
            for key in ["gamma", "tau"]:
                if key in optimized_data:
                    params[key] = optimized_data[key]

            param_rows.append(params)

    return {
        'ami_values': pd.DataFrame(ami_rows),
        'parameters': pd.DataFrame(param_rows)
    }


def format_params_string(row, method):
    """Format parameter string for display using hyperp macro."""
    if method in ["GSC-N", "GSC-UN"]:
        # Teleporting measure with t, alpha
        if 't' in row and 'alpha' in row:
            t = row['t']
            alpha = row['alpha']
            # Handle arrays by extracting first element
            if isinstance(t, (list, np.ndarray)):
                t = t[0] if len(t) > 0 else t
            if isinstance(alpha, (list, np.ndarray)):
                alpha = alpha[0] if len(alpha) > 0 else alpha
            # Convert to numeric types
            try:
                t = int(t)
                alpha = float(alpha)
                if pd.notna(t) and pd.notna(alpha):
                    return f" \\hyperp{{{t}, {alpha:.1f}}}"
            except (ValueError, TypeError):
                pass
    elif method == "DSC+":
        # DSC+ with gamma
        if 'gamma' in row:
            gamma = row['gamma']
            # Handle arrays by extracting first element
            if isinstance(gamma, (list, np.ndarray)):
                gamma = gamma[0] if len(gamma) > 0 else gamma
            # Convert to float
            try:
                gamma = float(gamma)
                if pd.notna(gamma):
                    return f" \\hyperp{{{gamma:.2f}}}"
            except (ValueError, TypeError):
                pass
    elif method in ["DI-SIM-R", "DI-SIM-L", "DI-SIM-C"]:
        # DI-SIM with tau
        if 'tau' in row:
            tau = row['tau']
            # Handle arrays by extracting first element
            if isinstance(tau, (list, np.ndarray)):
                tau = tau[0] if len(tau) > 0 else tau
            # Convert to float
            try:
                tau = float(tau)
                if pd.notna(tau):
                    return f" \\hyperp{{{tau:.2f}}}"
            except (ValueError, TypeError):
                pass

    return ""


def generate_competitors_table(results_path: str | Path, optimize_by: str = "ch",
                                dataset_order: list = None, output_file: str = None):
    """
    Generate LaTeX table comparing GSC methods with competitors.

    Parameters
    ----------
    results_path : str or Path
        Path to the benchmark_uci_grid_search results directory
    optimize_by : str
        Which metric to optimize by ("ch", "graph_ch", "ami")
    dataset_order : list, optional
        Order of datasets in the table
    output_file : str, optional
        Path to save the LaTeX table
    """
    # Load results
    results = load_results_with_params(results_path, optimize_by)
    ami_df = results['ami_values']
    params_df = results['parameters']

    if ami_df.empty:
        print(f"No results found for optimization by {optimize_by}")
        return

    # Define method order and their properties
    methods_config = [
        {"name": "SC-UN", "display": "SC-un", "show_params": False},
        {"name": "SC-N", "display": "SC-N", "show_params": False},
        {"name": "DSC+", "display": "DSC+", "show_params": True},
        {"name": "DI-SIM-R", "display": "DI-SIM-R", "show_params": True},
        {"name": "DI-SIM-L", "display": "DI-SIM-L", "show_params": True},
        {"name": "DI-SIM-C", "display": "DI-SIM-C", "show_params": True},
        {"name": "GSC-UN", "display": "GSC-un", "show_params": True},
        {"name": "GSC-N", "display": "GSC-N", "show_params": True},
    ]

    # Prepare data
    pivot_ami = ami_df.pivot(index="dataset", columns="method", values="ami")

    # Merge with parameters
    params_pivot = params_df.set_index(['dataset', 'method'])

    # Use provided dataset order or sort alphabetically
    if dataset_order:
        datasets = [d for d in dataset_order if d in pivot_ami.index]
    else:
        datasets = sorted(pivot_ami.index)

    # Start building LaTeX table
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"  \centering")

    optimize_label = "CH" if optimize_by == "ch" else "Graph-CH" if optimize_by == "graph_ch" else "AMI"

    lines.append(r"  \caption{\textbf{Comparison of clustering methods on UCI datasets.} " +
                f"AMI scores obtained when parameters are optimized for {optimize_label}. " +
                r"Optimized hyperparameters are shown using $\hyperp{\cdot}$.}")
    lines.append(r"  \label{tab:competitors_" + optimize_by + r"}")
    lines.append(r"  \begin{adjustbox}{width=\textwidth}")

    # Create table header with method columns
    n_methods = len(methods_config)
    col_spec = "l|" + "c" * n_methods
    lines.append(r"  \begin{tabular}{" + col_spec + r"}")
    lines.append(r"    \Xhline{2\arrayrulewidth}")

    # Header row with method names
    header_parts = ["Dataset"]
    for method_info in methods_config:
        header_parts.append(method_info["display"])

    header = "    " + " & ".join(header_parts) + r" \\"
    lines.append(header)
    lines.append(r"    \Xhline{2\arrayrulewidth}")

    # Data rows
    for dataset in datasets:
        row_parts = [dataset.replace('_', ' ').title()]
        row_values = []

        for method_info in methods_config:
            method = method_info["name"]

            if method in pivot_ami.columns and dataset in pivot_ami.index:
                ami_val = pivot_ami.loc[dataset, method]

                if pd.notna(ami_val):
                    # Get parameters if needed
                    param_str = ""
                    if method_info["show_params"]:
                        if (dataset, method) in params_pivot.index:
                            param_row = params_pivot.loc[(dataset, method)]
                            param_str = format_params_string(param_row, method)

                    cell_str = f"{ami_val:.3f}{param_str}"
                    row_values.append((ami_val, cell_str))
                else:
                    row_values.append((0.0, "--"))
            else:
                row_values.append((0.0, "--"))

        # Find best value(s) in this row
        if row_values:
            max_val = max(val for val, _ in row_values if val > 0)

            # Format row with best cells highlighted
            formatted_cells = []
            for val, cell_str in row_values:
                if val > 0 and abs(val - max_val) < 0.0001:  # Account for floating point
                    formatted_cells.append(f"\\bestcell{{{cell_str}}}")
                else:
                    formatted_cells.append(cell_str)

            row_parts.extend(formatted_cells)
            row_line = "    " + " & ".join(row_parts) + r" \\"
            lines.append(row_line)

    lines.append(r"    \Xhline{2\arrayrulewidth}")

    # Collect all methods in order
    all_methods = [m["name"] for m in methods_config]

    # Calculate competitiveness index
    competitiveness_indices = {}
    for method in all_methods:
        ratios = []
        for dataset in datasets:
            if method in pivot_ami.columns and dataset in pivot_ami.index:
                method_val = pivot_ami.loc[dataset, method]

                # Find best value for this dataset
                dataset_values = []
                for m in all_methods:
                    if m in pivot_ami.columns and dataset in pivot_ami.index:
                        val = pivot_ami.loc[dataset, m]
                        if pd.notna(val):
                            dataset_values.append(val)

                if dataset_values and pd.notna(method_val):
                    best_val = max(dataset_values)
                    if best_val > 0:
                        ratio = method_val / best_val
                        ratios.append(ratio)

        if ratios:
            competitiveness_indices[method] = np.mean(ratios)
        else:
            competitiveness_indices[method] = 0.0

    # Format competitiveness index row
    comp_parts = [r"\textit{Competitiveness}"]
    for method in all_methods:
        comp_parts.append(f"{competitiveness_indices.get(method, 0.0):.3f}")

    comp_line = "    " + " & ".join(comp_parts) + r" \\"
    lines.append(comp_line)

    # Calculate average ranks
    rank_data = []
    for dataset in datasets:
        dataset_values = {}
        for method in all_methods:
            if method in pivot_ami.columns and dataset in pivot_ami.index:
                val = pivot_ami.loc[dataset, method]
                if pd.notna(val):
                    dataset_values[method] = val

        if dataset_values:
            # Rank methods (higher AMI = better = lower rank number)
            sorted_methods = sorted(dataset_values.items(), key=lambda x: x[1], reverse=True)
            ranks = {}
            for rank, (method, _) in enumerate(sorted_methods, 1):
                ranks[method] = rank
            rank_data.append(ranks)

    # Calculate average ranks
    avg_ranks = {}
    for method in all_methods:
        method_ranks = [r[method] for r in rank_data if method in r]
        if method_ranks:
            avg_ranks[method] = np.mean(method_ranks)
        else:
            avg_ranks[method] = len(all_methods)

    # Format rank row
    rank_parts = [r"\textit{Avg. Rank}"]
    for method in all_methods:
        rank_parts.append(f"{avg_ranks.get(method, 0):.2f}")

    rank_line = "    " + " & ".join(rank_parts) + r" \\"
    lines.append(rank_line)
    lines.append(r"    \Xhline{2\arrayrulewidth}")

    lines.append(r"  \end{tabular}")
    lines.append(r"  \end{adjustbox}")
    lines.append(r"\end{table}")

    latex_table = "\n".join(lines)

    # Output
    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {output_file}")
    else:
        print(latex_table)

    return latex_table


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX table comparing GSC methods with competitors'
    )
    parser.add_argument('--results-path', type=str,
                       default='results/benchmark_uci_grid_search',
                       help='Path to results directory')
    parser.add_argument('--optimize-by', type=str, default='ch',
                       choices=['ch', 'graph_ch', 'ami'],
                       help='Metric used for optimization')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for LaTeX table')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Order of datasets in table')

    args = parser.parse_args()

    generate_competitors_table(
        results_path=args.results_path,
        optimize_by=args.optimize_by,
        dataset_order=args.datasets,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
