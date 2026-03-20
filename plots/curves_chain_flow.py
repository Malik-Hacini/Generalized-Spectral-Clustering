"""Plot AMI scores vs flow strength for the DiSBM-Chain benchmark."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.common import configure_paper_style, plot_method_lines, project_path, resolve_output_dir, summarize_mean_std, validate_selection

# Set to None to include all methods, or a list such as ["GSC-N", "SC-N", "DSC+"].
DEFAULT_METHODS_TO_PLOT = [
    "GSC-N",
    "SC-N",
    "DSC+",
    "DI-SIM-C",
    "DI-SIM-L",
    "DI-SIM-R"
]


def _parse_float_token(token: str) -> float:
    """Parse float tokens formatted as 0p1500 -> 0.1500."""
    return float(token.replace("p", "."))


def _extract_ami(best_results: dict) -> float | None:
    """Extract AMI from a best-results payload, preferring graph_ch-optimized AMI."""
    if "graph_ch" in best_results:
        payload = best_results["graph_ch"]
        if isinstance(payload, dict):
            ami_payload = payload.get("ami")
            if isinstance(ami_payload, dict) and "mean" in ami_payload:
                return float(ami_payload["mean"])

    if "ami" in best_results:
        payload = best_results["ami"]
        if isinstance(payload, dict):
            ami_payload = payload.get("ami")
            if isinstance(ami_payload, dict) and "mean" in ami_payload:
                return float(ami_payload["mean"])

    return None


def load_chain_flow_results(results_dir: str | Path) -> pd.DataFrame:
    """Load chain-flow benchmark rows from *_best_results.json files."""
    root = project_path(results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results directory not found: {root}")

    best_result_files = sorted(root.glob("*/*/*_best_results.json"))
    if not best_result_files:
        raise ValueError(f"No best_results.json files found in {root}")

    pattern = re.compile(
        r"^disbm_chainflow_b([0-9-]+)_pintra([0-9p]+)_pfwd([0-9p]+)_pbwd([0-9p]+)_seed(\d+)$"
    )

    rows: list[dict] = []
    unmatched_names: list[str] = []

    for best_file in best_result_files:
        dataset_name = best_file.parent.parent.name
        method = best_file.parent.name

        match = pattern.match(dataset_name)
        if match is None:
            if dataset_name.startswith("disbm_chainflow_"):
                unmatched_names.append(dataset_name)
            continue

        p_forward = _parse_float_token(match.group(3))
        p_backward = _parse_float_token(match.group(4))
        seed = int(match.group(5))

        best_results = json.loads(best_file.read_text())
        ami = _extract_ami(best_results)
        if ami is None:
            continue

        flow_ratio = np.nan if np.isclose(p_backward, 0.0) else p_forward / p_backward

        rows.append(
            {
                "method": method,
                "p_forward": p_forward,
                "p_backward": p_backward,
                "flow_ratio": flow_ratio,
                "seed": seed,
                "ami": ami,
            }
        )

    if unmatched_names:
        unique_unmatched = sorted(set(unmatched_names))
        print(
            "Warning: Could not parse "
            f"{len(unique_unmatched)} chain-flow dataset names. "
            f"First example: {unique_unmatched[0]}"
        )

    return pd.DataFrame(rows)


def plot_chain_flow_results(df: pd.DataFrame, output_file: Path, x_col: str) -> None:
    """Plot method AMI mean +/- std as a function of chain-flow strength."""
    summary = summarize_mean_std(df, ["method", x_col], "ami").sort_values(x_col)

    fig, ax = plt.subplots()
    plot_method_lines(ax, summary, x_col, "ami_mean", y_std_col="ami_std")

    if x_col == "flow_ratio":
        plt.xlabel(r"Flow Ratio ($p_{\mathrm{forward}} / p_{\mathrm{backward}}$)", fontsize=12)
    else:
        plt.xlabel(r"Forward Flow Strength ($\rho$)", fontsize=12)
    plt.ylabel("AMI", fontsize=12)
    plt.legend(loc="best", fontsize=10, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot results from chain-flow benchmark")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/benchmark_chain_flow_grid_search",
        help="Path to results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to plots/chain_flow/<experiment_name>/.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_METHODS_TO_PLOT,
        help="Methods to plot (default: all methods)",
    )
    parser.add_argument(
        "--x",
        type=str,
        choices=["p_forward", "flow_ratio"],
        default="p_forward",
        help="X-axis variable: forward flow probability or forward/backward ratio.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename. Defaults to chain_flow_ami_vs_<x>.pdf.",
    )
    args = parser.parse_args()
    configure_paper_style(plt)

    results_path = project_path(args.results_dir)
    output_dir = resolve_output_dir(args.output_dir, "chain_flow", results_path)

    print(f"Loading results from: {results_path}")
    df = load_chain_flow_results(results_path)
    if df.empty:
        print("No results found!")
        return

    print(f"Loaded {len(df)} result entries")
    all_methods = sorted(df["method"].unique())
    print(f"Methods: {all_methods}")

    selected_methods = validate_selection(all_methods, args.methods, "methods")
    df = df[df["method"].isin(selected_methods)].copy()
    print(f"Filtered to {len(selected_methods)} method(s): {selected_methods}")

    x_values = sorted(df[args.x].dropna().unique().tolist())
    print(f"{args.x} values: {x_values}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    output_name = args.output_name or f"chain_flow_ami_vs_{args.x}.pdf"
    output_file = output_dir / output_name
    plot_chain_flow_results(df=df, output_file=output_file, x_col=args.x)


if __name__ == "__main__":
    main()
