"""Proxy-based model selection utilities."""

from __future__ import annotations

import pandas as pd


def compute_dataset_selection(
    grid_long_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute oracle and proxy-selected AMI per dataset and proxy metric.

    Parameters
    ----------
    grid_long_df : pd.DataFrame
        Long-form GSC grid table with columns:
        ``dataset, gamma, seed, grid_index, alpha, t, ami, metric, metric_raw, metric_aligned``.
    baseline_df : pd.DataFrame
        Baseline table with columns ``dataset, sc_ami``.

    Returns
    -------
    pd.DataFrame
        One row per (dataset, metric) containing selected/oracle indices and
        derived AMI gaps.
    """
    required_grid_cols = {
        "dataset",
        "gamma",
        "seed",
        "grid_index",
        "alpha",
        "t",
        "ami",
        "metric",
        "metric_display",
        "metric_optimize",
        "metric_raw",
        "metric_aligned",
    }
    required_base_cols = {"dataset", "sc_ami"}

    missing_grid = required_grid_cols - set(grid_long_df.columns)
    missing_base = required_base_cols - set(baseline_df.columns)
    if missing_grid:
        raise ValueError(f"grid_long_df is missing required columns: {sorted(missing_grid)}")
    if missing_base:
        raise ValueError(f"baseline_df is missing required columns: {sorted(missing_base)}")

    baseline_lookup = baseline_df.set_index("dataset")["sc_ami"].to_dict()

    out_rows: list[dict] = []
    for (dataset, metric), subset in grid_long_df.groupby(["dataset", "metric"], sort=False):
        subset = subset.sort_values("grid_index")
        if subset.empty:
            continue

        if dataset not in baseline_lookup:
            raise KeyError(f"Missing SC baseline AMI for dataset '{dataset}'")

        idx_oracle = subset["ami"].idxmax()
        idx_proxy = subset["metric_aligned"].idxmax()

        row_oracle = subset.loc[idx_oracle]
        row_proxy = subset.loc[idx_proxy]
        sc_ami = float(baseline_lookup[dataset])

        ami_oracle = float(row_oracle["ami"])
        ami_selected = float(row_proxy["ami"])

        out_rows.append(
            {
                "dataset": dataset,
                "gamma": float(row_proxy["gamma"]),
                "seed": int(row_proxy["seed"]),
                "metric": str(metric),
                "metric_display": str(row_proxy["metric_display"]),
                "metric_optimize": str(row_proxy["metric_optimize"]),
                "sc_ami": sc_ami,
                "gsc_ami_oracle": ami_oracle,
                "gsc_ami_selected": ami_selected,
                "selection_regret": ami_oracle - ami_selected,
                "selected_minus_sc": ami_selected - sc_ami,
                "oracle_minus_sc": ami_oracle - sc_ami,
                "selected_grid_index": int(row_proxy["grid_index"]),
                "oracle_grid_index": int(row_oracle["grid_index"]),
                "selected_alpha": float(row_proxy["alpha"]),
                "selected_t": float(row_proxy["t"]),
                "oracle_alpha": float(row_oracle["alpha"]),
                "oracle_t": float(row_oracle["t"]),
                "selected_metric_raw": float(row_proxy["metric_raw"]),
                "selected_metric_aligned": float(row_proxy["metric_aligned"]),
                "oracle_metric_raw": float(row_oracle["metric_raw"]),
                "oracle_metric_aligned": float(row_oracle["metric_aligned"]),
                "n_grid_points": int(len(subset)),
            }
        )

    if not out_rows:
        raise RuntimeError("No dataset-level selection rows could be computed.")

    return pd.DataFrame(out_rows).sort_values(["metric", "gamma", "seed", "dataset"])
