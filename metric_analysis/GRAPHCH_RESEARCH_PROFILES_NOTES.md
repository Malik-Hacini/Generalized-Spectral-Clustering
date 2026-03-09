# Graph-CH Research Profile Notes


This note summarizes research-motivated diffusion filters for Graph-CH profile selection,
with references and implementation mapping.

## 1) Setting

Graph-CH uses distances between rows of

`Z = g(P) = sum_k a_k P^k`

where `P = D^{-1}A` is the row-stochastic random-walk matrix.

The profile design problem is to choose coefficients `a_k` so that:

- intra-cluster nodes are close in diffusion coordinates,
- inter-cluster nodes are far,
- induced Graph-CH correlates with supervised quality (AMI),
- selected `(alpha, t)` is close to AMI oracle.

## 2) Web-backed profile families

### A) Geometric / resolvent profiles

`a_k = rho^(k-1), 0 < rho < 1`

Motivation:

- Personalized PageRank is a geometric sum of random walk powers.
- This emphasizes short paths while retaining a controlled long-tail.

Reference:

- Andersen, Chung, Lang (2006), local graph partitioning with PageRank vectors.

### B) Heat-kernel (Poisson) profiles

`a_k = tau^k / k!` (truncated at `k <= K`, `k >= 1`)

Motivation:

- Heat kernel pagerank uses an exponential sum over random walk lengths.
- It is a diffusion-time smoothing with stronger attenuation of long, noisy paths.

Reference:

- Chung, Simpson (2015), heat kernel pagerank local clustering.

### C) Lazy-binomial profiles

`a_k = C(m,k)/2^m`, `k=1..m`

Motivation:

- `(I+P)/2` lazy walk suppresses periodicity artifacts.
- Binomial coefficients arise from `((I+P)/2)^m` expansion.

Reference context:

- Standard lazy random-walk formulations used in local clustering/PageRank analyses.

### D) Fejer/Cesaro smoothing profiles

`a_k = K+1-k`, `k=1..K`

Motivation:

- Triangular averaging smooths powers and reduces oscillatory effects.
- Emphasizes shorter-to-mid walk scales while still integrating over scales.

### E) Band-pass scale-contrast profiles

Example: `a_3 = 1, a_8 = -1`.

Motivation:

- Contrast between medium and long diffusion scales (difference of scales).
- Related to diffusion-wavelet ideas where differences between smoothed scales
  reveal mesoscopic structure.

Reference context:

- Spectral graph wavelet perspective (Hammond et al., 2011).

### F) User-specified custom profile

`P^3/3 + P^2/2` i.e. `a_2=1/2, a_3=1/3`.

Implemented profile id:

- `mix_p2_over2_p3_over3`

## 3) Implementation mapping in this repository

- Profile library: `utils/metrics/graph_CH/profiles.py`
- Legacy and research profile catalogs:
  - `build_legacy_graph_ch_profiles()`
  - `build_research_graph_ch_profiles()`
- Benchmark wiring:
  - `benchmark_networks_graphch_profiles.py`
  - env `GRAPHCH_PROFILE_SET=legacy|research|extended`
  - env `GRAPHCH_ONLY_PROFILES=<comma-separated profile_ids>`
- Fast profile re-evaluation from saved predictions:
  - `metric_analysis/evaluate_graphch_profiles_from_results.py`

## 4) Current lead-lag result snapshot (existing results tree)

Using:

`results/benchmark_lead_lag_graphch_profiles_grid_search`

The top regret/gain profile in the tested extended set is:

- `band_p3_minus_p8`

with effectively tied aggregate performance to:

- `delta_k03`

The custom requested profile:

- `mix_p2_over2_p3_over3`

underperforms `delta_k03` in this benchmark summary.

## 5) Recommended next experiment loop

1. Run profile-set `research` and `extended` benchmarks on lead-lag.
2. Evaluate with:

```bash
python -m metric_analysis.evaluate_graphch_profiles_from_results \
  --results-dir results/benchmark_lead_lag_graphch_profiles_grid_search \
  --profile-set extended
```

3. For top 3 profiles, generate per-year `(alpha,t)` AMI vs Graph-CH paired heatmaps.
4. Compare:
   - oracle hit count,
   - mean selection regret,
   - mean selected-minus-SC,
   - year-wise stability of selected `(alpha,t)`.
