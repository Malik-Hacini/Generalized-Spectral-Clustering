# DIGRAC Benchmarks and Analysis

This document covers the benchmark scripts and analysis commands for:
- DIGRAC synthetic DSBM families
- DIGRAC real directed networks

The analysis tooling is dataset-name agnostic and supports nested result folders
(for example, datasets saved under paths like `digrac_directed/<name>`).

## 1) Run DIGRAC DSBM family benchmarks

Run all DIGRAC synthetic families:

```bash
python3 benchmark_digrac_dsbm_types.py
```

Run selected families only:

```bash
DIGRAC_DSBM_TYPES=complete,cyclic python3 benchmark_digrac_dsbm_types.py
```

Optional: change the result-name prefix:

```bash
EXPERIMENT_PREFIX=benchmark_digrac_dsbm_v2 python3 benchmark_digrac_dsbm_types.py
```

Output pattern:

```text
results/<prefix>_<family>_graphch_profiles_grid_search
```

Default `<prefix>` is `benchmark_digrac_dsbm`.

## 2) Run network benchmarks (Graph-CH profiles + other metrics)

Graph-CH profile sweep:

```bash
python3 benchmark_networks_graphch_profiles.py
```

Graph-CH profile set controls:

```bash
# legacy profiles only (default)
GRAPHCH_PROFILE_SET=legacy python3 benchmark_networks_graphch_profiles.py

# research shortlist only (includes custom P^2/2 + P^3/3)
GRAPHCH_PROFILE_SET=research python3 benchmark_networks_graphch_profiles.py

# union of legacy + research profiles
GRAPHCH_PROFILE_SET=extended python3 benchmark_networks_graphch_profiles.py

# optional explicit subset by profile_id
GRAPHCH_PROFILE_SET=extended GRAPHCH_ONLY_PROFILES=delta_k03,mix_p2_over2_p3_over3 python3 benchmark_networks_graphch_profiles.py
```

Modularity + map-equation run:

```bash
python3 benchmark_networks_other_metrics.py
```

By default these scripts:
- include base network datasets (`email_eu_core`, `polblogs`)
- include DIGRAC directed labeled datasets from `datasets/digrac_directed/`
- unlabeled DIGRAC real datasets are not part of this benchmark set

Useful environment flags:

```bash
# run DIGRAC real networks only
INCLUDE_BASE_NETWORKS=0 INCLUDE_DIGRAC_NETWORKS=1 python3 benchmark_networks_graphch_profiles.py
INCLUDE_BASE_NETWORKS=0 INCLUDE_DIGRAC_NETWORKS=1 python3 benchmark_networks_other_metrics.py

# custom experiment directory names
EXPERIMENT_NAME=benchmark_networks_graphch_profiles_digrac python3 benchmark_networks_graphch_profiles.py
EXPERIMENT_NAME=benchmark_networks_other_metrics_digrac python3 benchmark_networks_other_metrics.py

# explicit dataset override (comma-separated)
NETWORK_DATASETS=digrac_directed/digrac_lead_lag_2001 python3 benchmark_networks_other_metrics.py
```

## 3) Analyze each DIGRAC DSBM family

`metric_analysis.cli` now auto-detects the axis from dataset names:
- `gamma` for `dsbm_gamma...`
- `eta` for DIGRAC DSBM names containing `eta...`
- fallback dataset index otherwise

For profile-swept Graph-CH runs, choose one profile via `--profile-id`.

```bash
PREFIX=benchmark_digrac_dsbm
PROFILE_ID=delta_k01

for FAMILY in complete cyclic star multipartite path; do
  python3 -m metric_analysis.cli \
    --results-dir "results/${PREFIX}_${FAMILY}_graphch_profiles_grid_search" \
    --proxy-metrics graph_ch modularity map_equation \
    --profile-id "${PROFILE_ID}" \
    --out-dir "results/${PREFIX}_${FAMILY}_graphch_profiles_grid_search/metric_analysis_${PROFILE_ID}"
done
```

## 4) Analyze real network benchmark family

Graph-CH profile analysis:

```bash
python3 -m metric_analysis.networks_graphch_profiles_analysis \
  --results-dir results/benchmark_networks_graphch_profiles_grid_search
```

Other metrics analysis:

```bash
python3 -m metric_analysis.networks_other_metrics_analysis \
  --results-dir results/benchmark_networks_other_metrics_grid_search
```

Best Graph-CH vs other proxies:

```bash
python3 -m metric_analysis.networks_compare_best_graphch \
  --graphch-analysis-dir results/benchmark_networks_graphch_profiles_grid_search/analysis_graphch_profiles \
  --other-analysis-dir results/benchmark_networks_other_metrics_grid_search/analysis_other_metrics \
  --out-dir results/benchmark_networks_metric_comparison
```

If you used custom `EXPERIMENT_NAME` values, pass matching `--results-dir` paths.

Evaluate new profile candidates on existing results (without rerunning clustering):

```bash
python3 -m metric_analysis.evaluate_graphch_profiles_from_results \
  --results-dir results/benchmark_lead_lag_graphch_profiles_grid_search \
  --profile-set extended
```
