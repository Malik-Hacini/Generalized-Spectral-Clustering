"""Plot runtime-vs-size line charts from runtime CSV outputs.

Expected input format is the runtime CSV produced by the experiment framework,
for example:
results/benchmark_runtimes_size_grid_search/benchmark_runtimes_size_runtimes.csv

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
import numpy as np
import pandas as pd

from plots.common import project_path, resolve_output_file, validate_selection


DEFAULT_RESULTS_CSV = (
    "results/benchmark_runtimes_size_grid_search/benchmark_runtimes_size_runtimes.csv"
)


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
        default="figures",
        help="Output directory for the figure.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="runtimes_size_lines.pdf",
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

    df = df[["dataset", "n"] + selected_methods].copy()
    df["n"] = pd.to_numeric(df["n"], errors="raise").astype(int)

    for m in selected_methods:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    return df, selected_methods


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
    colors = {
        "SC-UN": "#FF7E68",
        "GSC-UN": "#072AC8",
        "GSC-UN-NoTune": "#264DF7",
    }
    markers = {
        "SC-UN": "o",
        "GSC-UN": "s",
        "GSC-UN-NoTune": "^",
    }
    linestyles = {
        "SC-UN": "-",
        "GSC-UN": "--",
        "GSC-UN-NoTune": "-.",
    }
    display_names = {
        "SC-UN": "SC-UN",
        "GSC-UN": "GSC-UN",
        "GSC-UN-NoTune": "GSC-UN (w/o tuning)",
    }

    plt.figure()
    for method in methods:
        method_df = summary[summary["method"] == method]
        if method_df.empty:
            continue

        x = method_df["n"].to_numpy(dtype=float)
        y = method_df["runtime_median"].to_numpy(dtype=float)

        color = colors.get(method, None)
        marker = markers.get(method, "o")
        linestyle = linestyles.get(method, ":")
        plt.plot(
            x,
            y,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=6,
            color=color,
            label=display_names.get(method, method),
        )

    plt.xlabel(r"Number of points ($n$)")
    plt.ylabel(r"Runtime (s)")
    # plt.title(plt.title)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()


def main() -> None:
    args = parse_args()

    runtime_df, methods = load_runtime_table(args.results_csv, args.methods)
    summary = summarize_by_size(runtime_df, methods)

    plot_runtime_lines(
        summary=summary,
        methods=methods,
        title=args.title,
    )

    output_file = resolve_output_file(
        output_dir=args.output_dir,
        output_name=args.output_name,
        kind="figures",
        source=args.results_csv,
        default_name="runtimes_size_lines.pdf",
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved runtime line plot to: {output_file}")


if __name__ == "__main__":
    main()
