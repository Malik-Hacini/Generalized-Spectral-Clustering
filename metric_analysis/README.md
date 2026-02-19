# Metric Analysis

Utilities for AMI-proxy analysis of GSC grid-search results.

The package supports:
- DSBM proxy analysis (`graph_ch`, `modularity`, `map_equation`)
- merging split DSBM runs into one all-metrics tree
- network-specific analyses:
  - Graph-CH filter-profile comparison
  - modularity/map-equation comparison
  - best-Graph-CH-vs-others comparison

## DSBM workflow

If Graph-CH and (modularity/map-equation) were run separately, merge first:

```bash
python -m metric_analysis.merge_benchmark_runs \
  --graphch-root results/benchmark_dsbm_graphch_profiles_grid_search \
  --modmap-root results/benchmark_dsbm_grid_search \
  --output-root results/benchmark_dsbm_all_metrics_profiles_grid_search
```

Analyze with a single Graph-CH profile (required for profile-swept runs):

```bash
python -m metric_analysis.cli \
  --results-dir results/benchmark_dsbm_all_metrics_profiles_grid_search \
  --proxy-metrics graph_ch modularity map_equation \
  --profile-id delta_k01
```

## Network workflow

Run these analyses on finished benchmark outputs:

```bash
python -m metric_analysis.networks_graphch_profiles_analysis \
  --results-dir results/benchmark_networks_graphch_profiles_grid_search

python -m metric_analysis.networks_other_metrics_analysis \
  --results-dir results/benchmark_networks_other_metrics_grid_search

python -m metric_analysis.networks_compare_best_graphch \
  --graphch-analysis-dir results/benchmark_networks_graphch_profiles_grid_search/analysis_graphch_profiles \
  --other-analysis-dir results/benchmark_networks_other_metrics_grid_search/analysis_other_metrics \
  --out-dir results/benchmark_networks_metric_comparison
```

## Notes

- `map_equation` is minimized; aligned score is `-map_equation` for fair correlation comparisons.
- In profile-swept Graph-CH runs, pooling all profiles in one correlation is misleading; use `--profile-id`.
