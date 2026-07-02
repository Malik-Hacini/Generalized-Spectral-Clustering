"""Generate LaTeX table for AMI scores with different vertex measures."""

from __future__ import annotations

import argparse
from typing import cast
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from plots.common import load_best_result_entries, resolve_output_file


def _metric_candidates(metric: str) -> tuple[str, ...]:
    if metric == "graph_ch":
        return ("graph_ch", "ch")
    return (metric,)


def _resolve_available_metric(result_blob: dict, requested_metric: str) -> str | None:
    for candidate in _metric_candidates(requested_metric):
        if candidate in result_blob:
            return candidate
    return None


def _discover_available_opt_metrics(results_dir: str | Path) -> set[str]:
    available = set()
    for _, _, best_results in load_best_result_entries(results_dir):
        if "ami" in best_results:
            available.add("ami")
        if "ch" in best_results:
            available.add("ch")
        if "graph_ch" in best_results:
            available.add("graph_ch")
    return available


def _format_score_value(score: float, show_metric: str) -> str:
    return f"{score:.2f}"


def _optimize_label(optimize_by: str) -> str:
    if optimize_by == "ch":
        return "CH"
    if optimize_by == "ami":
        return "AMI"
    if optimize_by == "graph_ch":
        return "Graph-CH"
    return optimize_by.upper()


def _method_header(display: str, method: str, show_params: bool, optimize_by: str) -> str:
    base = rf"\textbf{{{display}}}"
    if not show_params:
        return base

    metric_label = _optimize_label(optimize_by)
    if method in {"GSC-N", "GSC-UN"}:
        return rf"{base}$\bestpar{{t,\alpha}}{{{metric_label}}}$"
    if method in {"deg-GSC-N", "deg-GSC-UN"}:
        return rf"{base}$\bestpar{{\gamma}}{{{metric_label}}}$"
    return base


def load_results_with_params(
    results_dir: str | Path,
    optimize_by: str = "ch",
    show_metric: str = "ami",
):
    score_rows = []
    param_rows = []
    for dataset_name, method_name, best_results in load_best_result_entries(results_dir):
        optimize_metric = _resolve_available_metric(best_results, optimize_by)
        if optimize_metric is None:
            continue
        optimized_data = best_results[optimize_metric]
        show_metric_resolved = _resolve_available_metric(optimized_data, show_metric)
        if show_metric_resolved and "mean" in optimized_data[show_metric_resolved]:
            score_rows.append(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "score": optimized_data[show_metric_resolved]["mean"],
                }
            )

        params = {"dataset": dataset_name, "method": method_name}
        measure = optimized_data.get("measure")
        if isinstance(measure, list) and len(measure) >= 2 and isinstance(measure[1], dict):
            params.update(measure[1])
        for key in ["gamma", "tau"]:
            if key in optimized_data:
                params[key] = optimized_data[key]
        param_rows.append(params)

    return pd.DataFrame(score_rows), pd.DataFrame(param_rows)


def format_params_string(row, method):
    if method in ["GSC-N", "GSC-UN"] and "t" in row and "alpha" in row:
        if pd.notna(row["t"]) and pd.notna(row["alpha"]):
            return f" \\hyperp{{{int(row['t'])}, {row['alpha']:.1f}}}"
    elif method in ["deg-GSC-N", "deg-GSC-UN"] and "gamma" in row and pd.notna(row["gamma"]):
        return f" \\hyperp{{{row['gamma']:.1f}}}"
    return ""


def generate_measures_table(
    results_path: str | Path,
    optimize_by: str = "ch",
    show_metric: str = "ami",
    dataset_order: list | None = None,
):
    score_df, params_df = load_results_with_params(results_path, optimize_by, show_metric)
    if score_df.empty:
        raise ValueError(
            f"No results found for optimization by {optimize_by} with metric {show_metric}"
        )

    method_groups = {
        "teleporting": {"methods": ["GSC-UN", "GSC-N"], "header": r"$\nu_{t,\alpha}$", "show_params": True},
        "degree": {"methods": ["deg-GSC-UN", "deg-GSC-N"], "header": r"$\nu_\textnormal{deg}(\gamma)$", "show_params": True},
        "uniform": {"methods": ["uniform-GSC-UN", "uniform-GSC-N"], "header": r"$\nu_\textnormal{unif}$", "show_params": False},
        "perron": {"methods": ["perron-GSC-UN", "perron-GSC-N"], "header": r"$\nu_\textnormal{Perron}$", "show_params": False},
    }

    pivot_scores = score_df.pivot(index="dataset", columns="method", values="score")
    params_pivot = params_df.set_index(["dataset", "method"])
    datasets = [d for d in dataset_order if d in pivot_scores.index] if dataset_order else sorted(pivot_scores.index)
    optimize_label = _optimize_label(optimize_by)
    show_label = _optimize_label(show_metric)

    lines = [
        r"\begin{table}",
        r"  \centering",
        r"  \caption{\textbf{Scores for different vertex measures.} "
        + f"For each dataset, we report the {show_label} score obtained by GSC with different vertex measures when parameters are optimized for {optimize_label}. "
        + r"Parameters are shown in parentheses: $\bestpar{t, \alpha}{"
        + optimize_label
        + r"}$ for $\nu_{t,\alpha}$ and $\bestpar{\gamma}{"
        + optimize_label
        + r"}$ for $\nu_\textnormal{deg}$.}",
        r"  \label{tab:measures_" + optimize_by + "_" + show_metric + r"}",
        r"  \begin{adjustbox}{width=\textwidth}",
        r"  \begin{tabular}{l|cc|cc|cc|cc}",
        r"    \Xhline{2\arrayrulewidth}",
    ]

    header1 = r"      \multirow{2}{*}{\textbf{Dataset}}"
    for group_info in method_groups.values():
        header1 += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{group_info['header']}}}}}"
    lines.append(header1 + r" \\")

    header2_parts = []
    for group_info in method_groups.values():
        for method in group_info["methods"]:
            display_name = method.replace("GSC-UN", r"GSC$_{\text{un}}$").replace("GSC-N", r"GSC$_{\text{n}}$")
            header2_parts.append(_method_header(display_name, method, group_info["show_params"], optimize_by))
    lines.append("    & " + " & ".join(header2_parts) + r" \\")
    lines.append(r"    \Xhline{2\arrayrulewidth}")

    for dataset in datasets:
        row_parts = [dataset.replace("_", " ").title()]
        row_values = []
        for group_info in method_groups.values():
            for method in group_info["methods"]:
                if method in pivot_scores.columns and dataset in pivot_scores.index and pd.notna(pivot_scores.loc[dataset, method]):
                    score_val = cast(float, pivot_scores.loc[dataset, method])
                    param_str = ""
                    if group_info["show_params"] and (dataset, method) in params_pivot.index:
                        param_str = format_params_string(params_pivot.loc[(dataset, method)], method)
                    row_values.append((score_val, f"{_format_score_value(score_val, show_metric)}{param_str}"))
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
            method: cast(float, pivot_scores.loc[dataset, method])
            for method in all_methods
            if method in pivot_scores.columns and dataset in pivot_scores.index and pd.notna(pivot_scores.loc[dataset, method])
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
        + " & ".join([r"\textit{PRB}"] + [f"{(sum(competitiveness[m]) / len(competitiveness[m])) if competitiveness[m] else 0.0:.2f}" for m in all_methods])
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
    parser.add_argument("--optimize-by", type=str, default=None, help="Metric used for optimization (ami, ch, graph_ch)")
    parser.add_argument("--show-metric", type=str, default=None, help="Metric displayed in table cells (ami, ch, graph_ch)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory. Defaults to plots/tables/<experiment_name>/.")
    parser.add_argument("--output-name", type=str, default=None, help="Output filename. Defaults to measures_<opt_metric>_show_<metric>.tex.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Order of datasets in table")
    args = parser.parse_args()

    valid_metrics = {"ami", "ch", "graph_ch"}
    if args.optimize_by is not None and args.optimize_by not in valid_metrics:
        raise ValueError("--optimize-by must be one of: ami, ch, graph_ch")
    if args.show_metric is not None and args.show_metric not in valid_metrics:
        raise ValueError("--show-metric must be one of: ami, ch, graph_ch")

    if (args.optimize_by is None) ^ (args.show_metric is None):
        raise ValueError("Provide both --optimize-by and --show-metric, or neither to generate default tables.")

    available_metrics = _discover_available_opt_metrics(args.results_dir)
    default_specs = [("ami", "ami")]
    if "graph_ch" in available_metrics:
        default_specs.extend([("graph_ch", "graph_ch"), ("graph_ch", "ami")])
    if "ch" in available_metrics:
        default_specs.extend([("ch", "ch"), ("ch", "ami")])
    default_specs = list(dict.fromkeys(default_specs))
    table_specs = [(args.optimize_by, args.show_metric)] if args.optimize_by else default_specs

    for optimize_by, show_metric in table_specs:
        output_file = resolve_output_file(
            args.output_dir,
            args.output_name if len(table_specs) == 1 else None,
            "tables",
            args.results_dir,
            f"measures_{optimize_by}_show_{show_metric}.tex",
        )
        latex_table = generate_measures_table(
            args.results_dir,
            optimize_by=optimize_by,
            show_metric=show_metric,
            dataset_order=args.datasets,
        )
        output_file.write_text(latex_table)
        print(f"LaTeX table saved to: {output_file}")


if __name__ == "__main__":
    main()
