# Metric Analysis

This package runs proxy-metric analysis for DSBM benchmark outputs.

It evaluates how well proxy metrics (e.g. `graph_ch`, `modularity`, `map_equation`)
track AMI on the GSC grid and how well they select hyperparameters `(alpha, t)`
without label access.

## Usage

Run on an existing grid-search benchmark directory:

```bash
source .venv/bin/activate
python -m metric_analysis.cli \
  --results-dir results/benchmark_dsbm_grid_search \
  --proxy-metrics graph_ch modularity map_equation
```

Custom output directory:

```bash
python -m metric_analysis.cli \
  --results-dir results/benchmark_dsbm_grid_search \
  --proxy-metrics graph_ch modularity map_equation \
  --out-dir results/benchmark_dsbm_grid_search/metric_analysis_all
```

## Output structure

- `manifests/run_config.json`: run metadata
- `tables/*.csv`: raw and aggregated statistics
- `figures/per_metric/<metric>/*.pdf`: metric-specific plots
- `figures/cross_metric/*.pdf`: cross-metric comparisons
- `index.md`: human-readable summary and reading order

## Notes

- For `map_equation` (minimize), the analysis also uses an aligned value (`-map_equation`)
  for fair correlation comparison against maximize-type metrics.
- Selection is done by maximizing aligned proxy value.
