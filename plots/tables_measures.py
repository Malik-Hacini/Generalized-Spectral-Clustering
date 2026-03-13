"""Generate LaTeX table for AMI scores with different vertex measures."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from plots.common import load_best_result_entries, resolve_output_file


def load_results_with_params(results_dir: str | Path, optimize_by: str = "ch"):
    ami_rows = []
    param_rows = []
    for dataset_name, method_name, best_results in load_best_result_entries(results_dir):
        if optimize_by not in best_results:
            continue
        optimized_data = best_results[optimize_by]
        if "ami" in optimized_data and "mean" in optimized_data["ami"]:
            ami_rows.append({"dataset": dataset_name, "method": method_name, "ami": optimized_data["ami"]["mean"]})

        params = {"dataset": dataset_name, "method": method_name}
        measure = optimized_data.get("measure")
        if isinstance(measure, list) and len(measure) >= 2 and isinstance(measure[1], dict):
            params.update(measure[1])
        for key in ["gamma", "tau"]:
            if key in optimized_data:
                params[key] = optimized_data[key]
        param_rows.append(params)

    return pd.DataFrame(ami_rows), pd.DataFrame(param_rows)


def format_params_string(row, method):
    if method in ["GSC-N", "GSC-UN"] and "t" in row and "alpha" in row:
        if pd.notna(row["t"]) and pd.notna(row["alpha"]):
            return f" \\hyperp{{{int(row['t'])}, {row['alpha']:.1f}}}"
    elif method in ["deg-GSC-N", "deg-GSC-UN"] and "gamma" in row and pd.notna(row["gamma"]):
        return f" \\hyperp{{{row['gamma']:.1f}}}"
    return ""


def generate_measures_table(results_path: str | Path, optimize_by: str = "ch", dataset_order: list | None = None):
    ami_df, params_df = load_results_with_params(results_path, optimize_by)
    if ami_df.empty:
        raise ValueError(f"No results found for optimization by {optimize_by}")

    method_groups = {
        "teleporting": {"methods": ["GSC-UN", "GSC-N"], "header": r"$\nu_{t,\alpha}$", "show_params": True},
        "degree": {"methods": ["deg-GSC-UN", "deg-GSC-N"], "header": r"$\nu_\textnormal{deg}(\gamma)$", "show_params": True},
        "uniform": {"methods": ["uniform-GSC-UN", "uniform-GSC-N"], "header": r"$\nu_\textnormal{unif}$", "show_params": False},
        "perron": {"methods": ["perron-GSC-UN", "perron-GSC-N"], "header": r"$\nu_\textnormal{Perron}$", "show_params": False},
    }

    pivot_ami = ami_df.pivot(index="dataset", columns="method", values="ami")
    params_pivot = params_df.set_index(["dataset", "method"])
    datasets = [d for d in dataset_order if d in pivot_ami.index] if dataset_order else sorted(pivot_ami.index)
    optimize_label = "CH" if optimize_by == "ch" else "Graph-CH" if optimize_by == "graph_ch" else "AMI"

    lines = [
        r"\begin{table}",
        r"  \centering",
        r"  \caption{\textbf{AMI scores for different vertex measures.} "
        + f"For each dataset, we report the AMI score obtained by GSC with different vertex measures when parameters are optimized for {optimize_label}. "
        + r"Parameters are shown in parentheses: $\bestpar{t, \alpha}{"
        + optimize_label
        + r"}$ for $\nu_{t,\alpha}$ and $\bestpar{\gamma}{"
        + optimize_label
        + r"}$ for $\nu_\textnormal{deg}$.}",
        r"  \label{tab:ami_measures_" + optimize_by + r"}",
        r"  \begin{adjustbox}{width=\textwidth}",
        r"  \begin{tabular}{l|cccccccc}",
        r"    \Xhline{2\arrayrulewidth}",
    ]

    header1 = r"      \multirow{2}{*}{Dataset}"
    for group_info in method_groups.values():
        header1 += f" & \\multicolumn{{2}}{{c}}{{{group_info['header']}}}"
    lines.append(header1 + r" \\")

    header2_parts = []
    for group_info in method_groups.values():
        for method in group_info["methods"]:
            display_name = method.replace("GSC-UN", "GSC-un").replace("GSC-N", "GSC-N")
            if group_info["show_params"]:
                param_str = r"$\bestpar{\gamma}{" + optimize_label + r"}$" if "deg" in method else r"$\bestpar{t, \alpha}{" + optimize_label + r"}$"
                display_name += f" {param_str}"
            header2_parts.append(display_name)
    lines.append("    & " + " & ".join(header2_parts) + r" \\")
    lines.append(r"    \Xhline{2\arrayrulewidth}")

    for dataset in datasets:
        row_parts = [dataset.replace("_", " ").title()]
        row_values = []
        for group_info in method_groups.values():
            for method in group_info["methods"]:
                if method in pivot_ami.columns and dataset in pivot_ami.index and pd.notna(pivot_ami.loc[dataset, method]):
                    ami_val = pivot_ami.loc[dataset, method]
                    param_str = ""
                    if group_info["show_params"] and (dataset, method) in params_pivot.index:
                        param_str = format_params_string(params_pivot.loc[(dataset, method)], method)
                    row_values.append((ami_val, f"{ami_val:.3f}{param_str}"))
                else:
                    row_values.append((0.0, "--"))

        max_val = max(val for val, _ in row_values if val > 0)
        for val, cell_str in row_values:
            row_parts.append(f"\\bestcell{{{cell_str}}}" if val > 0 and abs(val - max_val) < 1e-4 else cell_str)
        lines.append("  " + " & ".join(row_parts) + r" \\")

    lines.append(r"  \Xhline{2\arrayrulewidth}")
    all_methods = [method for group in method_groups.values() for method in group["methods"]]

    competitiveness = {method: [] for method in all_methods}
    ranks = {method: [] for method in all_methods}
    for dataset in datasets:
        dataset_values = {
            method: pivot_ami.loc[dataset, method]
            for method in all_methods
            if method in pivot_ami.columns and dataset in pivot_ami.index and pd.notna(pivot_ami.loc[dataset, method])
        }
        if not dataset_values:
            continue
        best_value = max(dataset_values.values())
        for method, value in dataset_values.items():
            competitiveness[method].append(value / best_value)
        for rank, (method, _) in enumerate(sorted(dataset_values.items(), key=lambda item: item[1], reverse=True), 1):
            ranks[method].append(rank)

    lines.append(
        "  "
        + " & ".join([r"\textit{Competitiveness}"] + [f"{(sum(competitiveness[m]) / len(competitiveness[m])) if competitiveness[m] else 0.0:.3f}" for m in all_methods])
        + r" \\")
    lines.append(
        "  "
        + " & ".join([r"\textit{Avg. Rank}"] + [f"{(sum(ranks[m]) / len(ranks[m])) if ranks[m] else len(all_methods):.2f}" for m in all_methods])
        + r" \\")
    lines.extend([r"  \Xhline{2\arrayrulewidth}", r"  \end{tabular}", r"  \end{adjustbox}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX table for AMI scores with different vertex measures")
    parser.add_argument("--results-dir", type=str, default="results/benchmark_uci_grid_search", help="Path to results directory")
    parser.add_argument("--optimize-by", type=str, default="ch", choices=["ch", "graph_ch", "ami"], help="Metric used for optimization")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory. Defaults to plots/tables/<experiment_name>/.")
    parser.add_argument("--output-name", type=str, default=None, help="Output filename. Defaults to measures_<metric>.tex.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Order of datasets in table")
    args = parser.parse_args()

    output_file = resolve_output_file(
        args.output_dir,
        args.output_name,
        "tables",
        args.results_dir,
        f"measures_{args.optimize_by}.tex",
    )
    latex_table = generate_measures_table(args.results_dir, args.optimize_by, args.datasets)
    output_file.write_text(latex_table)
    print(f"LaTeX table saved to: {output_file}")


if __name__ == "__main__":
    main()
