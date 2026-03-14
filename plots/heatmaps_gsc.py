"""Build GSC metric landscape heatmaps from grid-search result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from plots.common import configure_paper_style, project_path, resolve_output_dir, validate_selection


PALETTE = ["#072AC8", "#9A44C5", "#ff459c", "#F96C39"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GSC metric landscape heatmaps from grid-search results."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Experiment results directory, e.g. results/benchmark_uci_grid_search.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["AMI", "CH"],
        help="Metrics to plot.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to plot. Defaults to all datasets in the results folder.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to plot. Defaults to all methods found for each dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to plots/heatmaps/<experiment_name>/.",
    )
    return parser.parse_args()


def build_colormap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list("gsc_palette", PALETTE)
    cmap.set_bad("#D4D2D5")
    return cmap


def load_metric_grid(results_file: Path, metric_key: str):
    entries = json.loads(results_file.read_text())

    points = []
    t_values = set()
    alpha_values = set()
    for entry in entries:
        measure = entry["measure"][1]
        t = int(measure["t"])
        alpha = float(measure["alpha"])
        score = entry[metric_key]
        value = float(score["mean"] if isinstance(score, dict) else score)
        points.append((t, alpha, value))
        t_values.add(t)
        alpha_values.add(alpha)

    t_values = sorted(t_values)
    alpha_values = sorted(alpha_values)
    grid = np.full((len(alpha_values), len(t_values)), np.nan)

    t_index = {value: i for i, value in enumerate(t_values)}
    alpha_index = {value: i for i, value in enumerate(alpha_values)}
    for t, alpha, value in points:
        grid[alpha_index[alpha], t_index[t]] = value

    best_row, best_col = np.unravel_index(np.nanargmax(grid), grid.shape)
    return grid, t_values, alpha_values, t_values[best_col], alpha_values[best_row]


def plot_heatmap(grid, t_values, alpha_values, best_t, best_alpha, output_file: Path):
    fig, ax = plt.subplots(figsize=(8, 6))

    dt = 1 if len(t_values) < 2 else t_values[1] - t_values[0]
    da = 1 if len(alpha_values) < 2 else alpha_values[1] - alpha_values[0]
    extent = (
        t_values[0] - dt / 2,
        t_values[-1] + dt / 2,
        alpha_values[0] - da / 2,
        alpha_values[-1] + da / 2,
    )

    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        cmap=build_colormap(),
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.scatter(best_t, best_alpha, marker="*", s=320, c="white", edgecolors="white")
    fig.colorbar(image, ax=ax)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_paper_style(plt)
    results_dir = project_path(args.results_dir)
    output_dir = resolve_output_dir(args.output_dir, "heatmaps", results_dir)

    dataset_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    dataset_names = [path.name for path in dataset_dirs]
    selected_datasets = set(validate_selection(dataset_names, args.datasets, "datasets"))

    for dataset_dir in dataset_dirs:
        if dataset_dir.name not in selected_datasets:
            continue

        method_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
        method_names = [path.name for path in method_dirs]
        selected_methods = set(validate_selection(method_names, args.methods, "methods"))

        for method_dir in method_dirs:
            method_name = method_dir.name
            if method_name not in selected_methods:
                continue

            results_file = method_dir / f"{method_name}_all_results.json"
            if not results_file.exists():
                continue

            for metric in args.metrics:
                metric_upper = metric.upper()
                metric_lower = metric.lower()
                grid, t_values, alpha_values, best_t, best_alpha = load_metric_grid(results_file, metric_lower)
                output_file = (
                    output_dir
                    / metric_upper
                    / f"{dataset_dir.name}_{method_name}_t_alpha_{metric_lower}.pdf"
                )
                plot_heatmap(
                    grid,
                    t_values,
                    alpha_values,
                    best_t,
                    best_alpha,
                    output_file,
                )
                print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
