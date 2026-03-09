# Lead-Lag Graph-CH Analysis Guide

This guide explains how to analyze the lead-lag benchmark profile sweep (`delta_k*`, `prefix_k*`) and choose a Graph-CH profile using AMI-based diagnostics.

## 1) Run the profile benchmark (if not already done)

```bash
python3 benchmark_networks_graphch_profiles.py
```

For lead-lag only:

```bash
NETWORK_DATASETS=digrac_directed/digrac_lead_lag_2001,digrac_directed/digrac_lead_lag_2002,... \
EXPERIMENT_NAME=benchmark_lead_lag_graphch_profiles \
python3 benchmark_networks_graphch_profiles.py
```

## 2) Run Graph-CH profile analysis

```bash
.venv/bin/python -m metric_analysis.networks_graphch_profiles_analysis \
  --results-dir results/benchmark_lead_lag_graphch_profiles_grid_search
```

By default, outputs are written to:

`results/benchmark_lead_lag_graphch_profiles_grid_search/analysis_graphch_profiles`

## 3) Key outputs

### Tables

- `tables/profile_summary.csv`
  - Cross-profile ranking summary.
  - Main columns:
    - `overall_spearman`: profile-level proxy fidelity to AMI.
    - `mean_selection_regret`: AMI loss vs oracle (`AMI_oracle - AMI_selected`), lower is better.
    - `mean_selected_minus_sc`: selected GSC minus SC baseline, higher is better.

- `tables/profile_variability_summary.csv`
  - Variance/stability diagnostics for Graph-CH values by profile.
  - Main columns:
    - `log10_graph_ch_std`: global scale-stabilized dispersion.
    - `mean_within_dataset_cv`: average year-wise instability across `(alpha, t)` grid.
    - quality columns merged from `profile_summary.csv`.

- `tables/dataset_profile_metrics.csv`
  - Per-year/per-profile metrics: AMI selected, regret, gain vs SC, dataset-level Spearman.

- `tables/dataset_best_profiles.csv`
  - Per-year best profile by three criteria:
    - best AMI gain vs SC,
    - lowest regret,
    - highest within-year Spearman.

### Figures

- `figures/corr_vs_scale.pdf`
  - Pearson/Spearman vs profile scale for `delta_k` and `prefix_k`.

- `figures/regret_vs_scale.pdf`
  - Mean selection regret vs scale.

- `figures/selected_minus_sc_vs_scale.pdf`
  - Mean selected-minus-SC AMI gain vs scale.

- `figures/profile_stability_vs_quality.pdf`
  - Tradeoff view: stability (`log10` spread) vs proxy fidelity (Spearman), colored by regret.

- `figures/profile_variance_vs_scale.pdf`
  - Variance diagnostics vs scale (`log10` spread and within-year CV).

- `figures/heatmap_selected_minus_sc.pdf`
  - Year-by-profile AMI gain/loss relative to SC.

- `figures/heatmap_selection_regret.pdf`
  - Year-by-profile regret map (darker means larger oracle gap).

- `figures/heatmap_dataset_spearman.pdf`
  - Year-by-profile Spearman(Graph-CH, AMI).

## 4) How to choose a profile (recommended protocol)

Use a two-stage rule:

1. **Filter for proxy fidelity:** keep profiles with high `overall_spearman`.
2. **Select for utility/stability:** among those, pick low `mean_selection_regret` and low `log10_graph_ch_std` / `mean_within_dataset_cv`.

If your deployment objective is strictly AMI gain over SC, prioritize `mean_selected_minus_sc` and verify consistency with `heatmap_selected_minus_sc.pdf`.

## 5) Interpretation cautions

- High correlation does not guarantee best selected AMI.
- Some profiles can be numerically unstable (large Graph-CH dispersion) while still giving decent correlation.
- Prefer profiles that are both:
  - strong in aggregate metrics,
  - and not brittle across individual years in heatmaps.
