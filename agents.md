# Agents Guide: Generalized Spectral Clustering Research Repository

This repository is a math-first research codebase. Agents working here must behave like careful researchers, not just code generators.

## 1) Mission and Research Posture

- Primary goal: advance and validate Generalized Spectral Clustering (GSC) for directed and undirected graphs with mathematically defensible methods.
- Default stance: every non-trivial code decision should be tied to a formal objective, theorem-level intuition, or explicit experimental hypothesis.
- Treat correctness and rigor as first-class constraints; speed and convenience are secondary.
- Do not optimize only for benchmark numbers. Optimize for reproducible, interpretable, mathematically consistent conclusions.

## 2) Mandatory Reading Protocol

Before implementing or changing methodology, read the relevant papers in `papers/`.

### Required first read (highest priority)

- `papers/main_gsc/GDE_GSC.pdf`  
  This is the foundation of the project. You must align notation, objectives, and implementation choices with this paper.

### Core supporting theory (read when touching related components)

- `papers/von_luxburg_2007_spectral_clustering_tutorial.pdf` (spectral clustering foundations)
- `papers/coifman_lafon_2006_diffusion_maps.pdf` (diffusion geometry)
- `papers/fouss_etal_2007_random_walk_similarities.pdf` (random-walk graph similarities)
- `papers/leicht_newman_2008_directed_modularity.pdf` (directed modularity)
- `papers/newman_2013_spectral_modularity_equivalence.pdf`
- `papers/fasino_tudisco_2014_modularity_matrix_algebraic_analysis.pdf`
- `papers/fortunato_barthelemy_2007_modularity_resolution_limit.pdf`
- `papers/parPIC_parametrized_power_iteration_clustering.pdf`
- `papers/malliaros_vazirgiannis_2013_disim_directed_clustering.pdf`
- `papers/fan_etal_2019_hyperparameter_selection_spectral_clustering.pdf`

### Metrics and dataset-specific references

- `papers/metrics_datasets/notes.md`
- `papers/metrics_datasets/rosvall_bergstrom_map_equation_2008_arxiv.pdf`
- `papers/metrics_datasets/leicht_newman_directed_modularity_2008.pdf`
- `papers/metrics_datasets/delvenne_yaliraki_barahona_stability_2010.pdf`
- `papers/metrics_datasets/peixoto_hierarchical_block_structures_2014.pdf`

If your task touches a metric, include a short note in your reasoning about which reference justifies your implementation/interpretation.

## 3) Repository Map: What Is Canonical

### Core GSC implementation (authoritative)

- `scikit-learn/sklearn/cluster/_spectral.py`
  - Adds GSC-oriented API parameters: `standard`, `laplacian_method`, `measure`.
  - Resolves callable hyperparameters at fit time.
- `scikit-learn/sklearn/manifold/_spectral_embedding.py`
  - Embedding backend used by spectral clustering.
  - Handles `laplacian_method` variants and generalized Laplacian pathway.
- `scikit-learn/sklearn/manifold/_laplacian.py`
  - Generalized Laplacian construction.
- `scikit-learn/sklearn/utils/_param_validation.py`
  - Callable parameter resolver `_resolve_callable_param` used by configurable hyperparameters.

### Experiment framework (authoritative for benchmarks)

- `utils/experiments_utils.py`
  - Main experiment orchestration (`score`, `viz`, `grid_search`)
  - Method factory (`clusterer`)
  - Metric computation and grid expansion logic
- `utils/config.py`
  - Parameter hierarchy and method-specific parameter filtering
- `utils/file_manager.py`
  - Dataset load/save and result serialization

### Baselines and measure components

- `competitors/disim.py`
- `competitors/dsc.py`
- `competitors/measures.py`
- `competitors/neighbors.py`

### Graph-internal metrics

- `utils/metrics/modularity.py`
- `utils/metrics/map_equation.py`
- `utils/metrics/graph_CH/graph_ch.py`
- `utils/metrics/graph_CH/derivation.md`

### Data generation/import and benchmark scripts

- `benchmark_uci.py`
- `benchmark_dsbm.py`
- `benchmark_digrac_dsbm_types.py`
- `benchmark_networks_graphch_profiles.py`
- `benchmark_networks_other_metrics.py`
- `dsbm_utils/generate_dsbm_datasets.py`
- `dsbm_utils/dsbm_derivation.md`
- `data_import/import_digrac_data.py`

## 4) Mathematical Conventions to Preserve

Use notation and semantics consistent with the main GSC paper.

- Adjacency matrix: `A` (possibly directed)
- Transition matrix: `P = D^{-1} A`
- Generalized measure: `nu` and associated terms (`xi`, `nu + xi`)
- Laplacian families:
  - unnormalized
  - normalized
  - random-walk form
- GSC vs SC convention:
  - SC baseline: `standard=True`, often `measure=None`
  - GSC: `standard=False`, generalized measure active

When introducing a new formula or operator, explicitly state:

1. domain assumptions (directed/undirected, weighted/unweighted, connectedness),
2. normalization used,
3. optimization direction (maximize/minimize),
4. computational complexity class,
5. numerical stability strategy.

## 5) Core Code Patterns You Must Follow

### Parameter system pattern

- Respect the 4-level precedence in `ExperimentConfig`:
  1) `default_params`
  2) `dataset_params`
  3) `method_params`
  4) `method_dataset_params`
- Preserve callable parameter semantics: parameters can be literal values or `(callable, kwargs)` tuples resolved at runtime.
- Do not bypass `_resolve_callable_param` when adding new callable-capable parameters.

### Experiment spec pattern

- Methods are identified by `(implicit_name, explicit_name)` pairs.
- Keep naming stable for result compatibility (e.g., `SC-N`, `GSC-N`, `SC-UN`, `GSC-UN`).
- For repeated runs (`n_it`), preserve deterministic seed progression (`random_state + i`) unless there is a strong reason to change it.

### Metric pattern

- Keep metric orientation explicit:
  - maximize: `ami`, `ari`, `nmi`, `ch`, `modularity`, `graph_ch`
  - minimize: `map_equation`
- Maintain type checks and graceful behavior for invalid modality:
  - graph metrics on non-graph data should not silently produce misleading values.
- Keep outputs serializable and schema-stable (JSON/CSV contract used by analysis tooling).

### Sparse-first numerical pattern

- Prefer sparse operations for graph-scale computations.
- Avoid unnecessary dense conversion (`toarray`) except where numerically unavoidable and documented.
- Any dense fallback must include an explicit rationale and expected scale limitations.

## 6) Rigor Requirements for Any Research Change

For any algorithmic or metric change, include these elements in your working notes or PR summary:

1. **Hypothesis**: what mathematical behavior should improve, and why.
2. **Derivation sketch**: key equations or transformation steps.
3. **Assumptions**: graph class, ergodicity, reversibility, sparsity, etc.
4. **Failure modes**: where the method can break or become biased.
5. **Validation plan**: benchmarks/ablation and expected directional outcomes.

Do not ship "just because it improves one run". Require repeated evidence or principled justification.

## 7) Validation and Reproducibility Expectations

At minimum, validate with the most relevant benchmark script(s):

- `python benchmark_uci.py`
- `python benchmark_dsbm.py`
- `python benchmark_digrac_dsbm_types.py`
- `python benchmark_networks_graphch_profiles.py`
- `python benchmark_networks_other_metrics.py`

If the change targets complexity claims:

- `python complexity_benchmark_clean.py`

For DIGRAC workflows and analysis commands, follow `BENCHMARKS_DIGRAC.md`.

Reproducibility rules:

- report seeds and grid ranges,
- preserve result directory structure,
- keep method names and metric keys stable,
- do not alter metric definitions without documenting backward-compatibility impact.

## 8) Special Note on `metric_analysis/`

`metric_analysis/` is intentionally used for quick, practical analysis and does not fully match the architectural rigor of the core pipeline.

Treat it as:

- useful for rapid exploratory diagnostics,
- acceptable for quick proxy comparisons,
- non-canonical for core methodological truth.

If insights from `metric_analysis/` influence core algorithm decisions, re-validate those conclusions in the canonical benchmark pipeline before claiming research conclusions.

## 9) Scope Guardrails

- Do not treat the whole vendored `scikit-learn/` tree as open refactor territory.
- Limit changes to files relevant to GSC extensions unless absolutely necessary.
- Avoid API-breaking changes to benchmark outputs unless migration is explicit.
- Keep mathematical notation and terminology consistent with the main GSC paper.

## 10) Research-Grade Completion Checklist

Before considering a task complete, verify:

- [ ] You read the relevant paper(s), especially `papers/main_gsc/GDE_GSC.pdf`.
- [ ] Your change has a clear mathematical rationale.
- [ ] Assumptions and optimization direction are explicit.
- [ ] Results are reproducible with specified scripts/seeds.
- [ ] You did not rely solely on `metric_analysis/` for final claims.
- [ ] Outputs remain compatible with existing analysis tooling.

When in doubt, prioritize mathematical clarity, controlled experiments, and reproducibility over implementation novelty.
