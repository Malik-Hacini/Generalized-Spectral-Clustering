# Complexity Experiments

This folder contains small, readable scripts for validating the complexity derivations in `complexity_analysis/complexity.typ` and `complexity_analysis/final_results.typ`.

## Scripts

- `generate_gaussian_pointcloud.py`
  - Generates Gaussian point-cloud datasets and writes a manifest.
- `run_gaussian_pointcloud_suite.py`
  - Runs the full recommended Gaussian point-cloud benchmark campaign.
- `benchmark_graph_construction.py`
  - Benchmarks exact directed `k`-NN graph construction.
- `benchmark_measure_construction.py`
  - Benchmarks the GSC measure `nu`.
- `benchmark_xi_construction.py`
  - Benchmarks `xi = P^T nu`.
- `benchmark_laplacian_construction.py`
  - Benchmarks SC symmetrization, SC Laplacian assembly, and GSC Laplacian assembly.
- `benchmark_eigensolver.py`
  - Benchmarks the sparse spectral step on pre-built SC and GSC Laplacians.
- `benchmark_full_run.py`
  - Compares end-to-end single-run SC and GSC runtimes.
- `benchmark_network_full_run.py`
  - Compares end-to-end single-run SC and GSC runtimes on precomputed DISBM networks.
- `benchmark_network_preprocessing.py`
  - Compares SC and GSC preprocessing only, without spectral decomposition or `kmeans`.
- `benchmark_network_overhead.py`
  - Measures the GSC-specific preprocessing overhead on precomputed DISBM networks.
- `benchmark_network_component_share.py`
  - Measures preprocessing times against the isolated eigensolver times to show which step dominates.
- `plot_benchmarks.py`
  - Generates clean runtime/theory plots from the summary CSV files.
- `plot_network_sc_gsc.py`
  - Generates one focused SC-vs-GSC figure for the precomputed-network analysis.
- `plot_network_preprocessing.py`
  - Generates a preprocessing-only SC-vs-GSC figure against `m = nnz(W)`.
- `plot_network_overhead.py`
  - Generates the overhead-contribution figure versus `m = nnz(W)`.
- `plot_network_component_share.py`
  - Generates a component-dominance figure comparing preprocessing to the eigensolver.

## Output layout

- `data/`: saved Gaussian point clouds.
- `results/`: raw CSV, summary CSV, and JSON fit summaries.
- `figures/`: PNG/PDF plots generated from the summary CSV files.

## Common result schema

Each benchmark summary CSV uses the same columns so the plotting script stays generic:

- `benchmark`, `series`
- `x_name`, `x_value`
- runtime statistics: `runtime_mean`, `runtime_std`, `runtime_median`, `runtime_min`, `runtime_max`
- theory fields when available: `theory_term`, `theory_label`, `time_over_theory`
- experiment metadata such as `N`, `d`, `K`, `t`, `alpha`, `nnz`

## Typical usage

```bash
.venv/bin/python complexity_analysis/experiments/benchmark_graph_construction.py --algorithm auto
.venv/bin/python complexity_analysis/experiments/benchmark_measure_construction.py
.venv/bin/python complexity_analysis/experiments/benchmark_xi_construction.py
.venv/bin/python complexity_analysis/experiments/benchmark_laplacian_construction.py
.venv/bin/python complexity_analysis/experiments/benchmark_eigensolver.py
.venv/bin/python complexity_analysis/experiments/benchmark_full_run.py --input-type pointcloud
.venv/bin/python complexity_analysis/experiments/benchmark_network_full_run.py
.venv/bin/python complexity_analysis/experiments/benchmark_network_preprocessing.py
.venv/bin/python complexity_analysis/experiments/benchmark_network_overhead.py
.venv/bin/python complexity_analysis/experiments/benchmark_network_component_share.py
.venv/bin/python complexity_analysis/experiments/plot_benchmarks.py
```

One-command Gaussian point-cloud campaign:

```bash
.venv/bin/python complexity_analysis/experiments/run_gaussian_pointcloud_suite.py
```

## Notes

- The scripts import the vendored `scikit-learn` tree from this repository so the measurements follow the modified GSC/SC pipeline used by the project.
- Gaussian point clouds are generated so that the directed `K`-NN graph used by the benchmark is fully connected (with `K = ceil(factor log N)` from `--neighbors-factor`).
- The DISBM network scripts accept `--degree-factor`; when set, the block probabilities are scaled so that the expected degree is `Theta(log N)`, hence `m = Theta(N log N)`.
- The component scripts isolate the step they are meant to study: graph construction, measure computation, `xi`, or Laplacian assembly.
- The graph-construction script reports backend-dependent theory references; for `auto`, treat them as heuristic baselines, not exact complexity laws.
- The full-run script reports wall-clock comparisons without forcing a single theory term on the spectral stage, since that part is implementation-sensitive in the current shift-invert ARPACK path.
