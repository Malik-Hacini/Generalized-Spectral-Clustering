"""Plot runtime-vs-size line charts from runtime CSV outputs.

Expected input format is the runtime CSV produced by the experiment framework.

Columns:
        dataset, n, <method_1>, <method_2>, ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from plots.common import (
    configure_paper_style,
    plot_method_lines,
    project_path,
    resolve_output_file,
    validate_selection,
)


DEFAULT_RESULTS_CSV = "results/benchmark_runtimes_size_grid_search/benchmark_runtimes_size_runtimes.csv"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot runtime (seconds) vs dataset size (number of points)."
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=DEFAULT_RESULTS_CSV,
        help="Path to runtime CSV.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to include. Default: all methods in CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to plots/runtimes/<experiment_name>/.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="runtimes_size.pdf",
        help="Output filename.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title.",
    )
    return parser.parse_args()


def load_runtime_table(
    results_csv: str | Path, methods: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    csv_path = project_path(results_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Runtime CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"dataset", "n"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {sorted(required)}")

    all_methods = [c for c in df.columns if c not in {"dataset", "n"}]
    selected_methods = validate_selection(all_methods, methods, "methods")
    if not selected_methods:
        raise ValueError("No method columns selected.")

    df = pd.DataFrame(df[["dataset", "n"] + selected_methods].copy())
    df["n"] = pd.Series(pd.to_numeric(df["n"], errors="raise"), index=df.index, dtype="int64")

    for m in selected_methods:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    return pd.DataFrame(df), selected_methods


def summarize_by_size(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    agg_parts = []
    for method in methods:
        stats = df.groupby("n")[method].agg(["median", "count"]).reset_index()
        stats.columns = ["n", "runtime_median", "count"]
        stats["method"] = method
        agg_parts.append(stats)

    out = pd.concat(agg_parts, ignore_index=True)
    return out.sort_values(["method", "n"])


def plot_runtime_lines(summary: pd.DataFrame, methods: list[str], title: str) -> None:
    fig, ax = plt.subplots()
    plot_method_lines(ax, summary, "n", "runtime_median", methods=methods, legend_kwargs={})
    ax.set_xlabel(r"Number of points ($n$)")
    ax.set_ylabel(r"Runtime (s)")
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    configure_paper_style(plt)

    runtime_df, methods = load_runtime_table(args.results_csv, args.methods)
    summary = summarize_by_size(runtime_df, methods)

    fig = plot_runtime_lines(
        summary=summary,
        methods=methods,
        title=args.title,
    )

    output_file = resolve_output_file(
        output_dir=args.output_dir,
        output_name=args.output_name,
        kind="runtimes",
        source=args.results_csv,
        default_name="runtimes_size.pdf",
    )
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved runtime line plot to: {output_file}")


if __name__ == "__main__":
    main()
