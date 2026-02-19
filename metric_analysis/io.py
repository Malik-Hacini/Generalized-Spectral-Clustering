"""I/O utilities for loading benchmark artifacts into analysis tables."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from metric_analysis.specs import MetricSpec


DATASET_RE = re.compile(r"^dsbm_gamma(?P<gamma>\d+(?:\.\d+)?)_seed(?P<seed>\d+)$")


def iter_gamma_dataset_dirs(results_dir: Path):
    """Iterate over DSBM dataset directories sorted by (gamma, seed)."""
    rows = []
    for path in results_dir.iterdir():
        if not path.is_dir():
            continue
        match = DATASET_RE.match(path.name)
        if match is None:
            continue
        gamma = float(match.group("gamma"))
        seed = int(match.group("seed"))
        rows.append((gamma, seed, path))

    rows.sort(key=lambda x: (x[0], x[1]))
    for gamma, seed, path in rows:
        yield path, gamma, seed


def _parse_measure_params(entry: dict) -> tuple[float, float]:
    measure = entry.get("measure", [None, {}])
    kwargs = {}
    if isinstance(measure, list) and len(measure) >= 2 and isinstance(measure[1], dict):
        kwargs = measure[1]
    alpha = float(kwargs.get("alpha", np.nan))
    t = float(kwargs.get("t", np.nan))
    return alpha, t


def _extract_metric_value(entry: dict, metric_name: str) -> float:
    if metric_name not in entry:
        raise KeyError(metric_name)
    payload = entry[metric_name]
    if isinstance(payload, dict):
        if "mean" not in payload:
            raise ValueError(f"Metric payload for '{metric_name}' has no 'mean' key")
        return float(payload["mean"])
    return float(payload)


def _extract_summary_best(row: pd.Series, metric_name: str) -> float:
    best_col = f"{metric_name}_best_mean"
    mean_col = f"{metric_name}_mean_across_grid"

    if best_col in row.index and pd.notna(row[best_col]):
        return float(row[best_col])
    if mean_col in row.index and pd.notna(row[mean_col]):
        return float(row[mean_col])
    raise KeyError(f"Summary row does not contain '{best_col}' nor '{mean_col}'")


def _load_method_best_ami_from_all_results(dataset_dir: Path, method_name: str) -> float:
    all_results_path = dataset_dir / method_name / f"{method_name}_all_results.json"
    if not all_results_path.exists():
        raise FileNotFoundError(f"Missing method results: {all_results_path}")

    with all_results_path.open("r", encoding="utf-8") as f:
        all_results = json.load(f)
    if not all_results:
        raise RuntimeError(f"Empty method results in {all_results_path}")

    ami_values = [_extract_metric_value(entry, "ami") for entry in all_results]
    return float(np.max(np.asarray(ami_values, dtype=float)))


def load_grid_and_baselines(
    results_dir: Path,
    metric_specs: list[MetricSpec],
    gsc_method: str = "GSC-N",
    sc_method: str = "SC-N",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load GSC grid records and SC baseline AMI by dataset.

    Returns
    -------
    grid_long_df : pd.DataFrame
        Long-form table with one row per dataset x grid point x proxy metric.
    baseline_df : pd.DataFrame
        One row per dataset with SC baseline AMI.
    """
    if not metric_specs:
        raise ValueError("metric_specs must not be empty")

    grid_rows: list[dict] = []
    baseline_rows: list[dict] = []

    missing_metric_counter = {spec.name: 0 for spec in metric_specs}
    seen_metric_counter = {spec.name: 0 for spec in metric_specs}

    for dataset_dir, gamma, seed in iter_gamma_dataset_dirs(results_dir):
        summary_path = dataset_dir / f"{dataset_dir.name}_summary.csv"
        gsc_results_path = dataset_dir / gsc_method / f"{gsc_method}_all_results.json"

        if not gsc_results_path.exists():
            continue

        sc_ami = None
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            sc_rows = summary_df[summary_df["method"] == sc_method]
            if not sc_rows.empty:
                try:
                    sc_ami = _extract_summary_best(sc_rows.iloc[0], "ami")
                except KeyError:
                    sc_ami = None

        if sc_ami is None:
            sc_ami = _load_method_best_ami_from_all_results(dataset_dir, sc_method)

        baseline_rows.append(
            {
                "dataset": dataset_dir.name,
                "gamma": gamma,
                "seed": seed,
                "sc_ami": float(sc_ami),
            }
        )

        with gsc_results_path.open("r", encoding="utf-8") as f:
            gsc_all = json.load(f)
        if not gsc_all:
            continue

        for grid_index, entry in enumerate(gsc_all):
            try:
                ami_value = _extract_metric_value(entry, "ami")
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot parse AMI in {gsc_results_path} at grid index {grid_index}"
                ) from exc

            alpha, t = _parse_measure_params(entry)

            for spec in metric_specs:
                try:
                    metric_raw = _extract_metric_value(entry, spec.name)
                except KeyError:
                    missing_metric_counter[spec.name] += 1
                    continue

                grid_rows.append(
                    {
                        "dataset": dataset_dir.name,
                        "gamma": gamma,
                        "seed": seed,
                        "gsc_method": gsc_method,
                        "grid_index": grid_index,
                        "alpha": alpha,
                        "t": t,
                        "ami": float(ami_value),
                        "metric": spec.name,
                        "metric_display": spec.display_name,
                        "metric_optimize": spec.optimize,
                        "metric_raw": float(metric_raw),
                        "metric_aligned": spec.aligned_value(float(metric_raw)),
                    }
                )
                seen_metric_counter[spec.name] += 1

    if not grid_rows:
        raise RuntimeError(
            f"No GSC grid rows loaded from {results_dir}. "
            "Ensure grid-search benchmark results exist and include the requested metrics."
        )

    grid_long_df = pd.DataFrame(grid_rows).sort_values(
        ["metric", "gamma", "seed", "dataset", "grid_index"]
    )
    baseline_df = pd.DataFrame(baseline_rows).drop_duplicates(subset=["dataset"]).sort_values(
        ["gamma", "seed", "dataset"]
    )

    missing_msgs = []
    for metric_name, count in missing_metric_counter.items():
        if count > 0:
            missing_msgs.append(f"{metric_name}: {count} rows missing")
    if missing_msgs:
        msg = "; ".join(missing_msgs)
        print(f"Warning: some metrics were missing in grid entries and were skipped ({msg}).")

    unavailable = [metric for metric, seen_count in seen_metric_counter.items() if seen_count == 0]
    if unavailable:
        missing = ", ".join(sorted(unavailable))
        raise RuntimeError(
            "Requested proxy metrics are not present in the benchmark grid results: "
            f"{missing}. Re-run benchmark with these metrics enabled."
        )

    return grid_long_df, baseline_df
