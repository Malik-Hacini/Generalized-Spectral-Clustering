"""Plot SC-N vs GSC-N selected vs GSC-N oracle on the lead-lag suite."""

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
import pandas as pd

from plots.common import configure_paper_style, project_path, resolve_output_file, validate_selection

DEFAULT_RESULTS_DIR = "results/benchmark_lead_lag_grid_search"

CURVE_STYLES = {
    "SC-N": {"color": "#FF7E68", "marker": "o", "linestyle": "-", "label": "SC-N"},
    "GSC-N-selected": {"color": "#072AC8", "marker": "s", "linestyle": "-", "label": "GSC-N"},
    "GSC-N-oracle": {"color": "#072AC8", "marker": "^", "linestyle": "--", "label": "GSC-N (oracle)"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SC-N vs GSC-N selected vs GSC-N oracle across the lead-lag dataset suite."
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Path to the benchmark results directory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to include. Default: all lead-lag datasets in the results folder.",
    )
    parser.add_argument(
        "--selected-metric",
        default=None,
        help="Metric used for the selected benchmark curve. Default: auto-detect (graph_ch > ch > ami).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to plots/curves/<experiment_name>/.",
    )
    parser.add_argument(
        "--output-name",
        default="lead_lag_sc_n_vs_gsc_n.pdf",
        help="Output filename.",
    )
    parser.add_argument("--title", default=None, help="Optional plot title.")
    return parser.parse_args()



def _read_json(path: Path):
    return json.loads(path.read_text())



def _auto_metric(best_results: dict, selected_metric: str | None) -> str:
    if selected_metric is not None:
        if selected_metric not in best_results:
            raise ValueError(f"Metric '{selected_metric}' not found in {sorted(best_results)}")
        return selected_metric
    for metric in ("graph_ch", "ch", "ami"):
        if metric in best_results:
            return metric
    return next(iter(best_results))



def _extract_ami_from_best(best_results: dict, metric: str) -> float:
    metric_block = best_results[metric]
    ami_block = metric_block.get("ami")
    if not isinstance(ami_block, dict) or "mean" not in ami_block:
        raise ValueError(f"Missing AMI score in metric block '{metric}'")
    return float(ami_block["mean"])



def _extract_oracle_ami(all_results: list[dict]) -> float:
    return max(float(entry["ami"]["mean"]) for entry in all_results)



def _dataset_order_key(dataset_name: str):
    match = re.search(r"(\d{4})$", dataset_name)
    return (0, int(match.group(1))) if match else (1, dataset_name)



def load_lead_lag_curves(results_dir: str | Path, datasets: list[str] | None, selected_metric: str | None) -> tuple[pd.DataFrame, str]:
    results_path = project_path(results_dir)
    dataset_dirs = sorted(
        path for path in results_path.iterdir() if path.is_dir() and path.name.startswith("digrac_lead_lag_")
    )
    dataset_names = [path.name for path in dataset_dirs]
    selected_datasets = set(validate_selection(dataset_names, datasets, "datasets"))

    rows = []
    detected_metric = None
    for dataset_dir in dataset_dirs:
        if dataset_dir.name not in selected_datasets:
            continue

        sc_best = _read_json(dataset_dir / "SC-N" / "SC-N_best_results.json")
        gsc_best = _read_json(dataset_dir / "GSC-N" / "GSC-N_best_results.json")
        gsc_all = _read_json(dataset_dir / "GSC-N" / "GSC-N_all_results.json")

        metric = _auto_metric(gsc_best, selected_metric)
        if detected_metric is None:
            detected_metric = metric
        elif metric != detected_metric:
            raise ValueError(
                f"Inconsistent selected metrics across datasets: '{detected_metric}' vs '{metric}'"
            )

        rows.append(
            {
                "dataset": dataset_dir.name,
                "order_key": _dataset_order_key(dataset_dir.name),
                "SC-N": _extract_ami_from_best(sc_best, metric),
                "GSC-N-selected": _extract_ami_from_best(gsc_best, metric),
                "GSC-N-oracle": _extract_oracle_ami(gsc_all),
            }
        )

    if not rows:
        raise ValueError(f"No lead-lag datasets found in {results_path}")

    df = pd.DataFrame(rows).sort_values("order_key").reset_index(drop=True)
    return df.drop(columns=["order_key"]), detected_metric or (selected_metric or "unknown")



def plot_lead_lag_curves(df: pd.DataFrame, title: str | None) -> None:
    plt.figure()
    x = range(len(df))
    labels = [dataset.rsplit("_", 1)[-1] for dataset in df["dataset"]]

    for series_name, style in CURVE_STYLES.items():
        plt.plot(
            x,
            df[series_name].to_numpy(dtype=float),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=6,
            label=style["label"],
        )

    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel("AMI")
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend(loc="upper left", fontsize=9, frameon=False)
    plt.tight_layout()



def main() -> None:
    args = parse_args()
    configure_paper_style(plt)

    curve_df, metric = load_lead_lag_curves(args.results_dir, args.datasets, args.selected_metric)
    plot_lead_lag_curves(curve_df, args.title)

    output_file = resolve_output_file(
        output_dir=args.output_dir,
        output_name=args.output_name,
        kind="curves",
        source=args.results_dir,
        default_name="lead_lag_sc_n_vs_gsc_n.pdf",
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved curve plot to: {output_file}")
    print(f"Selected metric: {metric}")


if __name__ == "__main__":
    main()
