"""Plot method runtime comparisons from experiment runtime CSV files."""

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
from matplotlib.colors import to_hex

from plots.common import project_path, resolve_output_dir, validate_selection


DEFAULT_RESULTS_CSV = Path("results/benchmark_uci_grid_search/benchmark_uci_runtimes.csv")
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "<", ">", "h", "*"]
METHOD_COLORS = [
    "#FF7E68",  # SC UN
    "#FF6347", # SC N
    "#27A727", # DSC+
    "#FF579F", # DiSim C
    "#FD84B8", # DiSim R
    "#FFB0D2", # DiSim L
    "#405BD3", # GSC UN
    "#072AC8", # GSC N
    # --- additional colors
    "#8C564B",
    "#BCBD22",
]
LOG_COLLISION_THRESHOLD = 0.055
COLLISION_X_STEP = 0.09


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a runtime comparison plot from an experiment runtimes CSV."
    )
    parser.add_argument(
        "--results-csv",
        default=str(DEFAULT_RESULTS_CSV),
        help="Path to the runtimes CSV file.",
    )
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to include.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to include.")
    parser.add_argument(
        "--output-dir",
        default="../figures/runtimes/",
        help="Output directory. Defaults to ../figures/runtimes/.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename stem. Defaults to <experiment_name>_runtime_comparison.",
    )
    parser.add_argument("--title", default=None, help="Custom plot title.")
    return parser.parse_args()


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "figure.facecolor": "white",
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 10 / 72,
            "figure.constrained_layout.w_pad": 4 / 72,
            "figure.constrained_layout.hspace": 0.08,
            "figure.constrained_layout.wspace": 0.02,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "text.color": "#2F2840",
            "axes.facecolor": "white",
            "axes.edgecolor": "#8E84A8",
            "axes.labelcolor": "#2F2840",
            "axes.titlecolor": "#2F2840",
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.color": "#4E4464",
            "ytick.color": "#4E4464",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 15,
        }
    )


def _default_title(results_csv: Path) -> str:
    benchmark_name = results_csv.stem.replace("_runtimes", "")
    return f"{benchmark_name.replace('_', ' ').title()} Runtime Comparison"


def _default_output_stem(results_csv: Path, output_dir: Path, output_name: str | None) -> Path:
    benchmark_name = results_csv.stem.replace("_runtimes", "")
    stem = output_name if output_name is not None else f"{benchmark_name}_runtime_comparison"
    return output_dir / stem


def load_runtime_data(results_csv: Path, methods: list[str] | None, datasets: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if "dataset" not in df.columns:
        raise ValueError(f"Missing 'dataset' column in {results_csv}")
    if "n" not in df.columns:
        raise ValueError(f"Missing 'n' column in {results_csv}")

    available_methods = [column for column in df.columns if column not in {"dataset", "n"}]
    selected_methods = validate_selection(available_methods, methods, "methods")
    available_datasets = df["dataset"].tolist()
    selected_datasets = validate_selection(available_datasets, datasets, "datasets")

    df = df.set_index("dataset")
    dataset_sample_counts = df["n"].astype(int).to_dict()
    df = df[selected_methods]
    dataset_order = sorted(selected_datasets, key=lambda dataset_name: dataset_sample_counts[dataset_name])
    df = df.loc[dataset_order].reset_index()

    long_df = df.melt(
        id_vars="dataset",
        value_vars=selected_methods,
        var_name="method",
        value_name="runtime_seconds",
    ).dropna(subset=["runtime_seconds"])

    if long_df.empty:
        raise ValueError("No runtime values remain after filtering.")
    if (long_df["runtime_seconds"] <= 0).any():
        raise ValueError("All runtime values must be strictly positive for log-scale plotting.")

    long_df["n_samples"] = long_df["dataset"].map(dataset_sample_counts)
    long_df["dataset"] = pd.Categorical(long_df["dataset"], categories=dataset_order, ordered=True)
    long_df["method"] = pd.Categorical(long_df["method"], categories=selected_methods, ordered=True)
    return long_df.sort_values(["dataset", "method"])


def _method_colors(methods: list[str]) -> dict[str, str]:
    return {method: to_hex(METHOD_COLORS[i % len(METHOD_COLORS)]) for i, method in enumerate(methods)}


def _collision_offsets(log_values: np.ndarray) -> np.ndarray:
    if len(log_values) <= 1:
        return np.zeros(len(log_values), dtype=float)

    order = np.argsort(log_values)
    sorted_values = log_values[order]
    sorted_offsets = np.zeros(len(log_values), dtype=float)

    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] - sorted_values[end - 1] <= LOG_COLLISION_THRESHOLD:
            end += 1
        group_size = end - start
        if group_size > 1:
            sorted_offsets[start:end] = (
                np.arange(group_size, dtype=float) - (group_size - 1) / 2
            ) * COLLISION_X_STEP
        start = end

    offsets = np.zeros(len(log_values), dtype=float)
    offsets[order] = sorted_offsets
    return offsets


def plot_runtimes(long_df: pd.DataFrame, title: str):
    methods = list(long_df["method"].cat.categories)
    datasets = list(long_df["dataset"].cat.categories)
    counts_df = long_df[["dataset", "n_samples"]].drop_duplicates()
    dataset_sample_counts = dict(
        zip(
            counts_df["dataset"].astype(str).tolist(),
            counts_df["n_samples"].astype(int).tolist(),
            strict=True,
        )
    )
    dataset_labels = [
        f"{dataset.replace('_', ' ')}\n(n={dataset_sample_counts[str(dataset)]})"
        for dataset in datasets
    ]
    colors = _method_colors(methods)
    markers = {method: MARKERS[i % len(MARKERS)] for i, method in enumerate(methods)}

    fig_width = max(9.0, 1.15 * len(datasets) + 1.8)
    fig = plt.figure(figsize=(fig_width, 9), layout="constrained")
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 10])

    ax = subfigs[1].subplots()

    x_positions = np.arange(len(datasets), dtype=float)
    for i, x in enumerate(x_positions):
        if i % 2 == 0:
            ax.axvspan(x - 0.48, x + 0.48, color="#FBF9FE", zorder=0)
        ax.axvline(x, color="#ECE7F4", linewidth=0.8, zorder=0)

    labeled_methods = set()
    for base_x, dataset in zip(x_positions, datasets, strict=True):
        dataset_df = long_df[long_df["dataset"] == dataset].copy()
        runtime_values = np.asarray(dataset_df["runtime_seconds"], dtype=float)
        dataset_df["x"] = base_x + _collision_offsets(np.log10(runtime_values))

        for _, row in dataset_df.iterrows():
            method = str(row["method"])
            label = method if method not in labeled_methods else None
            ax.scatter(
                row["x"],
                row["runtime_seconds"],
                s=60,
                marker=markers[method],
                color=colors[method],
                edgecolors="white",
                linewidths=0.9,
                label=label,
                zorder=3,
            )
            labeled_methods.add(method)

    ax.set_yscale("log")
    ax.grid(axis="y", which="major", color="#D8D2E3", linewidth=0.9)
    ax.grid(axis="y", which="minor", color="#F1EDF7", linewidth=0.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dataset_labels, rotation=28, ha="right")
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    ax.set_xlabel("Datasets (sorted by sample count)")
    ax.set_ylabel("Runtime (seconds, log scale)")

    legend_cols = min(4, max(1, len(methods)))
    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D(
            [0], [0],
            marker=markers[method],
            color="w",
            markerfacecolor=colors[method],
            markersize=10,
            label=method,
        )
        for method in methods
    ]
    ax_leg = subfigs[0].subplots()
    ax_leg.axis("off")
    ax_leg.legend(
        handles=legend_handles,
        loc="center",
        ncols=legend_cols,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.4,
    )

    # fig.suptitle(title, y=1.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["bottom"].set_color("#8E84A8")
    ax.spines["left"].set_color("#8E84A8")
    ax.tick_params(axis="x", length=0)

    runtime_values = long_df["runtime_seconds"].to_numpy()
    ax.set_ylim(runtime_values.min() / 1.35, runtime_values.max() * 1.35)

    return fig


def main() -> None:
    args = parse_args()
    _configure_style()

    results_csv = project_path(args.results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    output_dir = resolve_output_dir(args.output_dir, "runtimes", results_csv)
    output_stem = _default_output_stem(results_csv, output_dir, args.output_name)

    long_df = load_runtime_data(results_csv, args.methods, args.datasets)
    title = args.title if args.title is not None else _default_title(results_csv)
    fig = plot_runtimes(long_df, title)

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=400)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
