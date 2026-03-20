"""Plot preprocessing and full runtimes together for the size benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from runtimes_size import load_runtime_table, summarize_by_size
else:
    from plots.runtimes_size import load_runtime_table, summarize_by_size

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from plots.common import configure_paper_style, resolve_output_file, style_for_method, validate_selection

DEFAULT_FULL_RESULTS_CSV = (
    "results/benchmark_runtimes_size_grid_search/benchmark_runtimes_size_runtimes.csv"
)
DEFAULT_PREPROCESSING_RESULTS_CSV = (
    "results/benchmark_runtimes_size_preprocessing/benchmark_runtimes_size_preprocessing_runtimes.csv"
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot preprocessing and full runtimes vs dataset size."
    )
    parser.add_argument("--full-results-csv", default=DEFAULT_FULL_RESULTS_CSV, help="Path to full runtime CSV.")
    parser.add_argument(
        "--preprocessing-results-csv",
        default=DEFAULT_PREPROCESSING_RESULTS_CSV,
        help="Path to preprocessing runtime CSV.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["SC-UN", "GSC-UN-NoTune"],
        help="Methods to include. Defaults to the single-run methods used in the complexity discussion.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to plots/runtimes/<experiment_name>/.",
    )
    parser.add_argument(
        "--output-name",
        default="runtimes_preprocessing_vs_full.pdf",
        help="Output filename.",
    )
    parser.add_argument("--title", default=None, help="Optional title.")
    return parser.parse_args()



def plot_preprocessing_vs_full(full_summary, preprocessing_summary, methods: list[str], title: str | None) -> None:
    plt.figure()
    for method in methods:
        style = style_for_method(method)

        full_df = full_summary[full_summary["method"] == method]
        if not full_df.empty:
            plt.plot(
                full_df["n"].to_numpy(dtype=float),
                full_df["runtime_median"].to_numpy(dtype=float),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                markersize=6,
                label=f"{style['label']} full",
            )

        preprocessing_df = preprocessing_summary[preprocessing_summary["method"] == method]
        if not preprocessing_df.empty:
            plt.plot(
                preprocessing_df["n"].to_numpy(dtype=float),
                preprocessing_df["runtime_median"].to_numpy(dtype=float),
                color=style["color"],
                marker=style["marker"],
                linestyle=":",
                linewidth=1.8,
                markersize=5,
                alpha=0.9,
                label=f"{style['label']} preprocessing",
            )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Number of nodes ($n$)")
    plt.ylabel(r"Runtime (s)")
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()



def main() -> None:
    args = parse_args()
    configure_paper_style(plt)

    full_df, full_methods = load_runtime_table(args.full_results_csv)
    preprocessing_df, preprocessing_methods = load_runtime_table(args.preprocessing_results_csv)
    common_methods = [method for method in full_methods if method in preprocessing_methods]
    methods = validate_selection(common_methods, args.methods, "methods")

    full_summary = summarize_by_size(full_df, methods)
    preprocessing_summary = summarize_by_size(preprocessing_df, methods)
    plot_preprocessing_vs_full(full_summary, preprocessing_summary, methods, args.title)

    output_file = resolve_output_file(
        output_dir=args.output_dir,
        output_name=args.output_name,
        kind="runtimes",
        source=args.full_results_csv,
        default_name="runtimes_preprocessing_vs_full.pdf",
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved preprocessing-vs-full runtime plot to: {output_file}")


if __name__ == "__main__":
    main()
