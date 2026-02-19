"""Statistical computations for proxy-metric analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _safe_corr_with_pvalues(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return Pearson/Spearman coefficients and p-values with safe NaN fallback."""
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan, np.nan, np.nan, np.nan

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)
    return float(pearson_r), float(pearson_p), float(spearman_rho), float(spearman_p)


def compute_grid_correlations(grid_long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute AMI-vs-proxy correlations overall, by gamma, and by dataset."""
    required = {"metric", "metric_display", "gamma", "dataset", "ami", "metric_raw", "metric_aligned"}
    missing = required - set(grid_long_df.columns)
    if missing:
        raise ValueError(f"grid_long_df is missing required columns: {sorted(missing)}")

    overall_rows = []
    by_gamma_rows = []
    by_dataset_rows = []

    for metric_name, metric_df in grid_long_df.groupby("metric", sort=False):
        display_name = str(metric_df["metric_display"].iloc[0])
        optimize = str(metric_df["metric_optimize"].iloc[0])

        pr_raw, pp_raw, sr_raw, sp_raw = _safe_corr_with_pvalues(
            metric_df["metric_raw"].to_numpy(dtype=float),
            metric_df["ami"].to_numpy(dtype=float),
        )
        pr_al, pp_al, sr_al, sp_al = _safe_corr_with_pvalues(
            metric_df["metric_aligned"].to_numpy(dtype=float),
            metric_df["ami"].to_numpy(dtype=float),
        )
        overall_rows.append(
            {
                "metric": metric_name,
                "metric_display": display_name,
                "metric_optimize": optimize,
                "n": int(len(metric_df)),
                "pearson_raw": pr_raw,
                "pearson_raw_p": pp_raw,
                "spearman_raw": sr_raw,
                "spearman_raw_p": sp_raw,
                "pearson_aligned": pr_al,
                "pearson_aligned_p": pp_al,
                "spearman_aligned": sr_al,
                "spearman_aligned_p": sp_al,
            }
        )

        for gamma, gamma_df in metric_df.groupby("gamma", sort=True):
            gpr_raw, gpp_raw, gsr_raw, gsp_raw = _safe_corr_with_pvalues(
                gamma_df["metric_raw"].to_numpy(dtype=float),
                gamma_df["ami"].to_numpy(dtype=float),
            )
            gpr_al, gpp_al, gsr_al, gsp_al = _safe_corr_with_pvalues(
                gamma_df["metric_aligned"].to_numpy(dtype=float),
                gamma_df["ami"].to_numpy(dtype=float),
            )
            by_gamma_rows.append(
                {
                    "metric": metric_name,
                    "metric_display": display_name,
                    "metric_optimize": optimize,
                    "gamma": float(gamma),
                    "n": int(len(gamma_df)),
                    "pearson_raw": gpr_raw,
                    "pearson_raw_p": gpp_raw,
                    "spearman_raw": gsr_raw,
                    "spearman_raw_p": gsp_raw,
                    "pearson_aligned": gpr_al,
                    "pearson_aligned_p": gpp_al,
                    "spearman_aligned": gsr_al,
                    "spearman_aligned_p": gsp_al,
                }
            )

        for dataset_name, ds_df in metric_df.groupby("dataset", sort=False):
            dpr_raw, dpp_raw, dsr_raw, dsp_raw = _safe_corr_with_pvalues(
                ds_df["metric_raw"].to_numpy(dtype=float),
                ds_df["ami"].to_numpy(dtype=float),
            )
            dpr_al, dpp_al, dsr_al, dsp_al = _safe_corr_with_pvalues(
                ds_df["metric_aligned"].to_numpy(dtype=float),
                ds_df["ami"].to_numpy(dtype=float),
            )
            by_dataset_rows.append(
                {
                    "metric": metric_name,
                    "metric_display": display_name,
                    "metric_optimize": optimize,
                    "dataset": dataset_name,
                    "gamma": float(ds_df["gamma"].iloc[0]),
                    "seed": int(ds_df["seed"].iloc[0]),
                    "n": int(len(ds_df)),
                    "pearson_raw": dpr_raw,
                    "pearson_raw_p": dpp_raw,
                    "spearman_raw": dsr_raw,
                    "spearman_raw_p": dsp_raw,
                    "pearson_aligned": dpr_al,
                    "pearson_aligned_p": dpp_al,
                    "spearman_aligned": dsr_al,
                    "spearman_aligned_p": dsp_al,
                }
            )

    overall_df = pd.DataFrame(overall_rows).sort_values("metric")
    by_gamma_df = pd.DataFrame(by_gamma_rows).sort_values(["metric", "gamma"])
    by_dataset_df = pd.DataFrame(by_dataset_rows).sort_values(["metric", "gamma", "seed", "dataset"])
    return overall_df, by_gamma_df, by_dataset_df


def summarize_metrics(
    selection_df: pd.DataFrame,
    overall_corr_df: pd.DataFrame,
    dataset_corr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build high-level metric ranking summary for proxy-model selection quality."""
    required_sel = {
        "metric",
        "metric_display",
        "gsc_ami_selected",
        "gsc_ami_oracle",
        "sc_ami",
        "selection_regret",
        "selected_minus_sc",
        "oracle_minus_sc",
    }
    missing_sel = required_sel - set(selection_df.columns)
    if missing_sel:
        raise ValueError(f"selection_df missing required columns: {sorted(missing_sel)}")

    rows = []
    for metric_name, subset in selection_df.groupby("metric", sort=False):
        display = str(subset["metric_display"].iloc[0])
        optimize = str(subset["metric_optimize"].iloc[0])

        corr_overall = overall_corr_df[overall_corr_df["metric"] == metric_name]
        if corr_overall.empty:
            raise RuntimeError(f"Missing overall correlation row for metric '{metric_name}'")
        corr_row = corr_overall.iloc[0]

        corr_dataset = dataset_corr_df[dataset_corr_df["metric"] == metric_name]

        rows.append(
            {
                "metric": metric_name,
                "metric_display": display,
                "metric_optimize": optimize,
                "n_datasets": int(len(subset)),
                "mean_gsc_selected_ami": float(subset["gsc_ami_selected"].mean()),
                "mean_gsc_oracle_ami": float(subset["gsc_ami_oracle"].mean()),
                "mean_sc_ami": float(subset["sc_ami"].mean()),
                "mean_selected_minus_sc": float(subset["selected_minus_sc"].mean()),
                "std_selected_minus_sc": float(subset["selected_minus_sc"].std(ddof=0)),
                "mean_oracle_minus_sc": float(subset["oracle_minus_sc"].mean()),
                "std_oracle_minus_sc": float(subset["oracle_minus_sc"].std(ddof=0)),
                "mean_selection_regret": float(subset["selection_regret"].mean()),
                "std_selection_regret": float(subset["selection_regret"].std(ddof=0)),
                "overall_pearson_raw": float(corr_row["pearson_raw"]),
                "overall_spearman_raw": float(corr_row["spearman_raw"]),
                "overall_pearson_aligned": float(corr_row["pearson_aligned"]),
                "overall_spearman_aligned": float(corr_row["spearman_aligned"]),
                "mean_dataset_pearson_aligned": float(corr_dataset["pearson_aligned"].mean()),
                "mean_dataset_spearman_aligned": float(corr_dataset["spearman_aligned"].mean()),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        ["mean_selected_minus_sc", "overall_spearman_aligned", "mean_selection_regret"],
        ascending=[False, False, True],
    )
    return summary_df
