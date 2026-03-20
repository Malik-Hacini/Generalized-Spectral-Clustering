"""Plot AMI scores vs flow strength for the DiSBM-Chain benchmark."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.common import project_path, resolve_output_dir
from plots.method_style import ordered_methods, style_for_method

# Set to None to include all methods, or a list such as ["GSC-N", "SC-N", "DSC+"].
DEFAULT_METHODS_TO_PLOT = None


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
    summary = df.groupby(["method", x_col])["ami"].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["method", x_col, "ami_mean", "ami_std", "n"]
    summary = summary.sort_values(x_col)

    method_order = ordered_methods(summary["method"].unique().tolist())

    plt.figure()
    for method in method_order:
        method_data = summary[summary["method"] == method]
        if method_data.empty:
            continue

        style = style_for_method(method)
        x_values = np.asarray(method_data[x_col], dtype=float)
        ami_mean = np.asarray(method_data["ami_mean"], dtype=float)
        ami_std = np.nan_to_num(np.asarray(method_data["ami_std"], dtype=float), nan=0.0)

        plt.plot(
            x_values,
            ami_mean,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=6,
            linewidth=2,
            alpha=1,
        )
        plt.fill_between(x_values, ami_mean - ami_std, ami_mean + ami_std, color=style["color"], alpha=0.2)

    if x_col == "flow_ratio":
        plt.xlabel(r"Flow Ratio ($p_{\\mathrm{forward}} / p_{\\mathrm{backward}}$)", fontsize=12)
    else:
        plt.xlabel(r"Forward Flow Strength ($p_{\\mathrm{forward}}$)", fontsize=12)
    plt.ylabel("AMI Score", fontsize=12)
    plt.legend(loc="best", fontsize=10, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
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
    args = parser.parse_args()

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

    if args.methods is not None:
        selected_methods = [m for m in args.methods]
        missing_methods = set(selected_methods) - set(all_methods)
        if missing_methods:
            print(f"Warning: Methods not found in results: {sorted(missing_methods)}")
        available_selected = [m for m in selected_methods if m in all_methods]
        if not available_selected:
            print("Error: No selected methods found in results!")
            return
        df = df[df["method"].isin(available_selected)].copy()
        print(f"Filtered to {len(available_selected)} method(s): {available_selected}")

    x_values = sorted(df[args.x].dropna().unique().tolist())
    print(f"{args.x} values: {x_values}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    output_file = output_dir / f"chain_flow_ami_vs_{args.x}.pdf"
    plot_chain_flow_results(df=df, output_file=output_file, x_col=args.x)


if __name__ == "__main__":
    main()
