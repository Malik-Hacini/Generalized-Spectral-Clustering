"""Plot Gaussian-injection benchmark AMI mean +/- std.

Expected dataset naming pattern:
    gaussian_inj_n480_k8_bw1p0000_sigma0p0100_alpha0p0000_seed0
"""

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

from plots.method_style import ordered_methods, style_for_method


# Default parameters: edit here for your usual plotting setup.
DEFAULT_RESULTS_PATH = "results/benchmark_gaussian_injection_alpha_sigma_grid_search"
DEFAULT_OUTPUT_DIR = "figures"
DEFAULT_OPTIMIZE_BY = "graph_ch"
DEFAULT_INCLUDE_GSC_AMI_SUPERVISED = False
DEFAULT_FIXED_SIGMA = 0.5
DEFAULT_FIXED_ALPHA = 0.5
DEFAULT_SHOW_STD = True
# Set to None to include all methods, or provide a list to filter plotted methods.
DEFAULT_METHODS_TO_PLOT = [
  "GSC-N",
  "SC-N",
  "DSC+"
]


def _resolve_input_path(path_value: str | Path) -> Path:
    """Resolve paths robustly for runs launched outside the repository root."""
    path = Path(path_value)
    if path.exists() or path.is_absolute():
        return path

    repo_root_candidate = Path(__file__).resolve().parents[1]
    repo_relative = repo_root_candidate / path
    if repo_relative.exists():
        return repo_relative

    return path


def _parse_float_token(token: str) -> float:
    """Parse float tokens formatted as 1p2300 -> 1.2300."""
    return float(token.replace("p", "."))


def _extract_ami_value(best_results: dict, optimize_by: str = "graph_ch") -> float | None:
    """Extract AMI score from a best-results payload for a chosen optimization metric."""
    metric_payload = best_results.get(optimize_by)
    if not isinstance(metric_payload, dict):
        return None

    ami_payload = metric_payload.get("ami")
    if isinstance(ami_payload, dict) and "mean" in ami_payload:
        return float(ami_payload["mean"])
    if isinstance(ami_payload, (int, float)):
        return float(ami_payload)

    linked_ami = metric_payload.get("linked_ami_mean")
    if isinstance(linked_ami, (int, float)):
        return float(linked_ami)

    return None


def load_gaussian_injection_results(
    results_path: str | Path,
    optimize_by: str = "graph_ch",
    include_gsc_ami_supervised: bool = True,
) -> pd.DataFrame:
    """Load per-dataset AMI values from `*_best_results.json` files.

    Returns rows with columns:
        method, sigma, alpha, seed, ami
    """
    results_dir = _resolve_input_path(results_path)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    best_result_files = sorted(results_dir.glob("*/*/*_best_results.json"))
    if not best_result_files:
        raise ValueError(f"No *_best_results.json files found in {results_dir}")

    pattern = re.compile(
        r"^gaussian_inj_n\d+_k\d+_bw[0-9p]+_sigma([0-9p]+)_alpha([0-9p]+)_seed(\d+)$"
    )

    rows: list[dict] = []
    unmatched_dataset_names: list[str] = []

    for best_file in best_result_files:
        method_name = best_file.parent.name
        dataset_name = best_file.parent.parent.name

        match = pattern.match(dataset_name)
        if match is None:
            if dataset_name.startswith("gaussian_inj_"):
                unmatched_dataset_names.append(dataset_name)
            continue

        sigma_value = _parse_float_token(match.group(1))
        alpha_value = _parse_float_token(match.group(2))
        seed_value = int(match.group(3))

        with best_file.open("r") as handle:
            best_results = json.load(handle)

        ami_value = _extract_ami_value(best_results, optimize_by=optimize_by)
        if ami_value is None:
            continue

        rows.append(
            {
                "method": method_name,
                "sigma": sigma_value,
                "alpha": alpha_value,
                "seed": seed_value,
                "ami": ami_value,
            }
        )

        # Optionally add a supervised AMI-optimized curve for GSC-N only.
        if include_gsc_ami_supervised and method_name == "GSC-N":
            ami_supervised_value = _extract_ami_value(best_results, optimize_by="ami")
            if ami_supervised_value is not None:
                rows.append(
                    {
                        "method": "GSC-N (AMI-opt)",
                        "sigma": sigma_value,
                        "alpha": alpha_value,
                        "seed": seed_value,
                        "ami": ami_supervised_value,
                    }
                )

    if unmatched_dataset_names:
        unique_unmatched = sorted(set(unmatched_dataset_names))
        print(
            "Warning: Could not parse "
            f"{len(unique_unmatched)} Gaussian-injection dataset names. "
            f"First example: {unique_unmatched[0]}"
        )

    return pd.DataFrame(rows)


def _plot_mean_std_lines(
    summary: pd.DataFrame,
    x_col: str,
    output_file: Path,
    title: str,
    xlabel: str,
    log_x: bool = False,
    show_std: bool = True,
) -> None:
    """Plot AMI mean line with std shading for each method."""
    method_order = ordered_methods(summary["method"].unique().tolist())

    fig, ax = plt.subplots(figsize=(8, 6))

    for method in method_order:
        method_data = summary[summary["method"] == method].sort_values(x_col)
        if method_data.empty:
            continue

        style = style_for_method(method)
        x_values = np.asarray(method_data[x_col], dtype=float)
        ami_mean = np.asarray(method_data["ami_mean"], dtype=float)
        ami_std = np.nan_to_num(np.asarray(method_data["ami_std"], dtype=float), nan=0.0)

        ax.plot(
            x_values,
            ami_mean,
            label=style.get("label", method),
            color=style.get("color", None),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            linewidth=2.5,
            markersize=6,
        )
        if show_std:
            ax.fill_between(
                x_values,
                ami_mean - ami_std,
                ami_mean + ami_std,
                color=style.get("color", None),
                alpha=0.2,
            )
            ax.set_ylabel("AMI (mean +/- std)", fontsize=12)
        else:
            ax.set_ylabel("AMI (mean)", fontsize=12)

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel, fontsize=12)

    # ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=11)
    plt.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Gaussian-injection AMI mean +/- std from graph_ch-optimized benchmark results."
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=DEFAULT_RESULTS_PATH,
        help="Path to Gaussian-injection benchmark results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures are saved.",
    )
    parser.add_argument(
        "--optimize-by",
        type=str,
        default=DEFAULT_OPTIMIZE_BY,
        help="Metric used for selecting best params in JSON (default: graph_ch).",
    )
    parser.add_argument(
        "--gsc-ami-supervised",
        dest="gsc_ami_supervised",
        action="store_true",
        default=DEFAULT_INCLUDE_GSC_AMI_SUPERVISED,
        help="Enable plotting GSC-N AMI selected by supervised AMI optimization.",
    )
    parser.add_argument(
        "--no-gsc-ami-supervised",
        dest="gsc_ami_supervised",
        action="store_false",
        help="Disable plotting GSC-N AMI selected by supervised AMI optimization.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS_TO_PLOT,
        help="Methods to plot. Default comes from DEFAULT_METHODS_TO_PLOT.",
    )
    parser.add_argument(
        "--fixed-sigma",
        type=float,
        default=DEFAULT_FIXED_SIGMA,
        help="Sigma value used to plot AMI vs alpha.",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=DEFAULT_FIXED_ALPHA,
        help="Alpha value used to plot AMI vs sigma.",
    )
    parser.add_argument(
        "--show-std",
        dest="show_std",
        action="store_true",
        default=DEFAULT_SHOW_STD,
        help="Show mean +/- std shaded bands (default: on).",
    )
    parser.add_argument(
        "--no-show-std",
        dest="show_std",
        action="store_false",
        help="Disable std shading and show mean curves only.",
    )
    args = parser.parse_args()

    output_dir = _resolve_input_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.results_path}")
    df = load_gaussian_injection_results(
        args.results_path,
        optimize_by=args.optimize_by,
        include_gsc_ami_supervised=args.gsc_ami_supervised,
    )

    if df.empty:
        print("No entries found after parsing results.")
        return

    if args.methods is not None:
        available_methods = sorted(df["method"].unique())
        selected_methods = [method for method in args.methods if method in available_methods]
        missing_methods = [method for method in args.methods if method not in available_methods]
        if missing_methods:
            print(f"Warning: requested methods not found and skipped: {missing_methods}")
        if not selected_methods:
            raise ValueError(
                "No selected methods were found in parsed results. "
                f"Available methods: {available_methods}"
            )
        df = df[df["method"].isin(selected_methods)].copy()

    print(f"Loaded {len(df)} entries")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    tol = 1e-12

    # AMI vs alpha for fixed sigma
    df_alpha = df[np.isclose(df["sigma"], args.fixed_sigma, atol=tol)]
    if df_alpha.empty:
        print(f"No rows found for sigma={args.fixed_sigma}. Skipping alpha plot.")
    else:
        summary_alpha = (
            df_alpha.groupby(["method", "alpha"])["ami"]
            .agg(ami_mean="mean", ami_std="std", n="count")
            .reset_index()
        )

        alpha_out = output_dir / (
            f"gaussian_injection_ami_mean_std_vs_alpha_sigma{args.fixed_sigma:.4f}_{args.optimize_by}.pdf"
        )
        _plot_mean_std_lines(
            summary=summary_alpha,
            x_col="alpha",
            output_file=alpha_out,
            title=f"AMI vs Injection Alpha (sigma={args.fixed_sigma}, optimize={args.optimize_by})",
            xlabel="Injection alpha (blending weight)",
            log_x=False,
            show_std=args.show_std,
        )

    # AMI vs sigma for fixed alpha
    df_sigma = df[np.isclose(df["alpha"], args.fixed_alpha, atol=tol)]
    if df_sigma.empty:
        print(f"No rows found for alpha={args.fixed_alpha}. Skipping sigma plot.")
    else:
        summary_sigma = (
            df_sigma.groupby(["method", "sigma"])["ami"]
            .agg(ami_mean="mean", ami_std="std", n="count")
            .reset_index()
        )

        sigma_out = output_dir / (
            f"gaussian_injection_ami_mean_std_vs_sigma_alpha{args.fixed_alpha:.4f}_{args.optimize_by}.pdf"
        )
        _plot_mean_std_lines(
            summary=summary_sigma,
            x_col="sigma",
            output_file=sigma_out,
            title=f"AMI vs Injected Sigma (alpha={args.fixed_alpha}, optimize={args.optimize_by})",
            xlabel="Injected sigma",
            log_x=True,
            show_std=args.show_std,
        )


if __name__ == "__main__":
    main()
