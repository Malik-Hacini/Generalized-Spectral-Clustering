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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.common import configure_paper_style, plot_method_lines, project_path, resolve_output_dir, summarize_mean_std, validate_selection


# Default parameters: edit here for your usual plotting setup.
DEFAULT_RESULTS_DIR = "results/benchmark_gaussian_injection_alpha_sigma_grid_search"
DEFAULT_OUTPUT_DIR = None
DEFAULT_OPTIMIZE_BY = "graph_ch"
DEFAULT_INCLUDE_GSC_AMI_SUPERVISED = False
DEFAULT_FIXED_SIGMA = 0.8
DEFAULT_FIXED_ALPHA = 0.5
DEFAULT_SHOW_STD = True
# Set to None to include all methods, or provide a list to filter plotted methods.
DEFAULT_METHODS_TO_PLOT = [
  "GSC-N",
  "SC-N",
  "DSC+"
]

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
    results_dir: str | Path,
    optimize_by: str = "graph_ch",
    include_gsc_ami_supervised: bool = True,
) -> pd.DataFrame:
    """Load per-dataset AMI values from `*_best_results.json` files.

    Returns rows with columns:
        method, sigma, alpha, seed, ami
    """
    results_dir = project_path(results_dir)
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
    show_legend: bool = True,
) -> None:
    """Plot AMI mean line with std shading for each method."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x_values = np.asarray(summary[x_col], dtype=float)
    plot_method_lines(
        ax,
        summary.sort_values(x_col),
        x_col,
        "ami_mean",
        y_std_col="ami_std" if show_std else None,
        show_legend=show_legend,
        legend_kwargs={"loc": "best", "fontsize": 11} if show_legend else None,
    )
    ax.set_ylabel("AMI (mean +/- std)" if show_std else "AMI (mean)", fontsize=12)

    if log_x:
        ax.set_xscale("log")
        positive_x = np.sort(np.unique(x_values[x_values > 0]))
        if positive_x.size > 0:
            # Keep bounds tight around observed values to avoid empty decade padding.
            if positive_x.size == 1:
                ax.set_xlim(positive_x[0] / 1.2, positive_x[0] * 1.2)
            else:
                ax.set_xlim(positive_x[0] / 1.08, positive_x[-1] * 1.08)
            ax.set_xticks(positive_x)
    elif x_values.size > 0:
        unique_x = np.sort(np.unique(x_values))
        if unique_x.size == 1:
            ax.set_xlim(unique_x[0] - 0.1, unique_x[0] + 0.1)
        else:
            span = unique_x[-1] - unique_x[0]
            ax.set_xlim(unique_x[0] - 0.04 * span, unique_x[-1] + 0.04 * span)

    ax.set_xlabel(xlabel, fontsize=12)

    # ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
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
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Path to Gaussian-injection benchmark results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory. Defaults to plots/imbalance/<experiment_name>/.",
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

    configure_paper_style(plt)
    output_dir = resolve_output_dir(args.output_dir, "imbalance", args.results_dir)

    print(f"Loading results from: {args.results_dir}")
    df = load_gaussian_injection_results(
        args.results_dir,
        optimize_by=args.optimize_by,
        include_gsc_ami_supervised=args.gsc_ami_supervised,
    )

    if df.empty:
        print("No entries found after parsing results.")
        return

    if args.methods is not None:
        available_methods = sorted(df["method"].astype(str).unique().tolist())
        selected_methods = validate_selection(available_methods, args.methods, "methods")
        df = df[df["method"].isin(selected_methods)].copy()

    print(f"Loaded {len(df)} entries")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    tol = 1e-12

    # AMI vs alpha for fixed sigma
    df_alpha = pd.DataFrame(df[np.isclose(df["sigma"], args.fixed_sigma, atol=tol)]).copy()
    if df_alpha.empty:
        print(f"No rows found for sigma={args.fixed_sigma}. Skipping alpha plot.")
    else:
        summary_alpha = summarize_mean_std(pd.DataFrame(df_alpha), ["method", "alpha"], "ami")

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
            show_legend=False,
        )

    # AMI vs sigma for fixed alpha
    df_sigma = pd.DataFrame(df[np.isclose(df["alpha"], args.fixed_alpha, atol=tol)]).copy()
    if df_sigma.empty:
        print(f"No rows found for alpha={args.fixed_alpha}. Skipping sigma plot.")
    else:
        summary_sigma = summarize_mean_std(pd.DataFrame(df_sigma), ["method", "sigma"], "ami")

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
            show_legend=True,
        )


if __name__ == "__main__":
    main()
