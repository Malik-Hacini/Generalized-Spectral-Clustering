"""Generate LaTeX table comparing GSC methods with competitors."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from plots.common import load_best_result_entries, resolve_output_file


def _metric_candidates(metric: str) -> tuple[str, ...]:
    """Return ordered metric keys to try for loading results."""
    if metric == "graph_ch":
        return ("graph_ch", "ch")
    return (metric,)


def _resolve_available_metric(result_blob: dict, requested_metric: str) -> str | None:
    """Resolve metric key with fallbacks (graph_ch -> ch)."""
    for candidate in _metric_candidates(requested_metric):
        if candidate in result_blob:
            return candidate
    return None


def _discover_available_opt_metrics(results_dir: str | Path) -> set[str]:
    """Discover which optimization metrics are present in result files."""
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
    """Format score with 2 decimals.

    AMI is displayed on [0, 1] scale (0.xx), while CH/Graph-CH remain raw values.
    """
    if show_metric == "ami":
        return f"{score:.2f}"
    return f"{score:.2f}"


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
        try:
            return f" \\hyperp{{{int(row['t'])}, {float(row['alpha']):.1f}}}"
        except (ValueError, TypeError):
            return ""
    if method == "DSC+" and "gamma" in row:
        try:
            return f" \\hyperp{{{float(row['gamma']):.2f}}}"
        except (ValueError, TypeError):
            return ""
    if method in ["DI-SIM-R", "DI-SIM-L", "DI-SIM-C"] and "tau" in row:
        try:
            return f" \\hyperp{{{float(row['tau']):.2f}}}"
        except (ValueError, TypeError):
            return ""
    return ""


def _optimize_label(optimize_by: str) -> str:
    if optimize_by == "ch":
        return "CH"
    if optimize_by == "ami":
        return "AMI"
    if optimize_by == "graph_ch":
        return "Graph-CH"
    return optimize_by.upper()


def _method_header(display: str, method: str, show_params: bool, optimize_by: str) -> str:
    """Build method header, adding bestpar annotation for parameterized methods."""
    base = rf"\textbf{{{display}}}"
    if not show_params:
        return base

    metric_label = _optimize_label(optimize_by)
    if method in {"GSC-N", "GSC-UN"}:
        symbol = r"t,\alpha"
        base = (
            r"\textbf{GSC$_{\text{n}}$}"
            if method == "GSC-N"
            else r"\textbf{GSC$_{\text{un}}$}"
        )
    elif method == "DSC+":
        symbol = r"\gamma"
    elif method in {"DI-SIM-R", "DI-SIM-L", "DI-SIM-C"}:
        symbol = r"\tau"
    else:
        return base

    return rf"{base}$\bestpar{{{symbol}}}{{{metric_label}}}$"


def generate_competitors_table(
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

    methods_config = [
        {"name": "SC-UN", "display": r"SC$_{\text{un}}$", "show_params": False},
        {"name": "SC-N", "display": r"SC$_{\text{N}}$", "show_params": False},
        {"name": "DSC+", "display": r"DSC+", "show_params": True},
        {"name": "DI-SIM-R", "display": r"DI-SIM$_{\text{R}}$", "show_params": True},
        {"name": "DI-SIM-L", "display": r"DI-SIM$_{\text{L}}$", "show_params": True},
        {"name": "DI-SIM-C", "display": r"DI-SIM$_{\text{C}}$", "show_params": True},
        {"name": "GSC-UN", "display": r"GSC$_{\text{un}}$", "show_params": True},
        {"name": "GSC-N", "display": r"GSC$_{\text{N}}$", "show_params": True},
    ]

    # Add a visual split between non-GSC baselines and GSC variants.
    gsc_start_idx = next(
        (i for i, method in enumerate(methods_config) if method["name"].startswith("GSC")),
        len(methods_config),
    )
    if 0 < gsc_start_idx < len(methods_config):
        column_spec = "l|" + "c" * gsc_start_idx + "|" + "c" * (len(methods_config) - gsc_start_idx)
    else:
        column_spec = "l|" + "c" * len(methods_config)

    pivot_scores = score_df.pivot(index="dataset", columns="method", values="score")
    params_pivot = params_df.set_index(["dataset", "method"])
    datasets = [d for d in dataset_order if d in pivot_scores.index] if dataset_order else sorted(pivot_scores.index)
    optimize_label = "CH" if optimize_by == "ch" else "Graph-CH" if optimize_by == "graph_ch" else "AMI"
    show_label = _optimize_label(show_metric)

    header_labels = [
        _method_header(m["display"], m["name"], m["show_params"], optimize_by)
        for m in methods_config
    ]

    lines = [
        r"\begin{table}",
        r"  \centering",
        r"  \caption{\textbf{Comparison of clustering methods on UCI datasets.} "
        + f"{show_label} scores obtained when parameters are optimized for {optimize_label}. "
        + r"Optimized hyperparameters are shown using $\hyperp{\cdot}$.}",
        r"  \label{tab:competitors_" + optimize_by + "_" + show_metric + r"}",
        r"  \begin{adjustbox}{width=\textwidth}",
        r"  \begin{tabular}{" + column_spec + r"}",
        r"    \Xhline{2\arrayrulewidth}",
        "    " + " & ".join(["Dataset"] + header_labels) + r" \\",
        r"    \Xhline{2\arrayrulewidth}",
    ]

    for dataset in datasets:
        row_parts = [dataset.replace("_", " ").title()]
        row_values = []
        for method_info in methods_config:
            method = method_info["name"]
            if method in pivot_scores.columns and dataset in pivot_scores.index and pd.notna(pivot_scores.loc[dataset, method]):
                score_val = pivot_scores.loc[dataset, method]
                param_str = ""
                if method_info["show_params"] and (dataset, method) in params_pivot.index:
                    param_str = format_params_string(params_pivot.loc[(dataset, method)], method)
                row_values.append((score_val, f"{_format_score_value(score_val, show_metric)}{param_str}"))
            else:
                row_values.append((0.0, "--"))

        max_val = max(val for val, _ in row_values if val > 0)
        for val, cell_str in row_values:
            row_parts.append(f"\\bestcell{{{cell_str}}}" if val > 0 and abs(val - max_val) < 1e-4 else cell_str)
        lines.append("    " + " & ".join(row_parts) + r" \\")

    lines.append(r"    \Xhline{2\arrayrulewidth}")
    all_methods = [m["name"] for m in methods_config]

    competitiveness = {method: [] for method in all_methods}
    ranks = {method: [] for method in all_methods}
    for dataset in datasets:
        dataset_values = {
            method: pivot_scores.loc[dataset, method]
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
        "    "
        + " & ".join([r"\textit{Competitiveness}"] + [f"{(sum(competitiveness[m]) / len(competitiveness[m])) if competitiveness[m] else 0.0:.2f}" for m in all_methods])
        + r" \\")
    lines.append(
        "    "
        + " & ".join([r"\textit{Avg. Rank}"] + [f"{(sum(ranks[m]) / len(ranks[m])) if ranks[m] else len(all_methods):.2f}" for m in all_methods])
        + r" \\")
    lines.extend([r"    \Xhline{2\arrayrulewidth}", r"  \end{tabular}", r"  \end{adjustbox}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX table comparing GSC methods with competitors")
    parser.add_argument("--results-dir", type=str, default="results/benchmark_uci_grid_search", help="Path to results directory")
    parser.add_argument("--optimize-by", type=str, default=None, help="Metric used for optimization (ami, ch, graph_ch)")
    parser.add_argument("--show-metric", type=str, default=None, help="Metric displayed in table cells (ami, ch, graph_ch)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory. Defaults to plots/tables/<experiment_name>/.")
    parser.add_argument("--output-name", type=str, default=None, help="Output filename. Defaults to competitors_<opt_metric>_show_<metric>.tex.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Order of datasets in table")
    args = parser.parse_args()

    valid_metrics = {"ami", "ch", "graph_ch"}
    if args.optimize_by is not None and args.optimize_by not in valid_metrics:
        raise ValueError("--optimize-by must be one of: ami, ch, graph_ch")
    if args.show_metric is not None and args.show_metric not in valid_metrics:
        raise ValueError("--show-metric must be one of: ami, ch, graph_ch")

    if (args.optimize_by is None) ^ (args.show_metric is None):
        raise ValueError("Provide both --optimize-by and --show-metric, or neither to generate default 3 tables.")

    available_metrics = _discover_available_opt_metrics(args.results_dir)
    default_specs = [("ami", "ami")]
    if "graph_ch" in available_metrics:
        default_specs.extend([("graph_ch", "graph_ch"), ("graph_ch", "ami")])
    if "ch" in available_metrics:
        default_specs.extend([("ch", "ch"), ("ch", "ami")])

    # Keep order stable while removing duplicates.
    default_specs = list(dict.fromkeys(default_specs))
    table_specs = [(args.optimize_by, args.show_metric)] if args.optimize_by else default_specs

    for optimize_by, show_metric in table_specs:
        output_file = resolve_output_file(
            args.output_dir,
            args.output_name if len(table_specs) == 1 else None,
            "tables",
            args.results_dir,
            f"competitors_{optimize_by}_show_{show_metric}.tex",
        )
        latex_table = generate_competitors_table(
            args.results_dir,
            optimize_by=optimize_by,
            show_metric=show_metric,
            dataset_order=args.datasets,
        )
        output_file.write_text(latex_table)
        print(f"LaTeX table saved to: {output_file}")


if __name__ == "__main__":
    main()
