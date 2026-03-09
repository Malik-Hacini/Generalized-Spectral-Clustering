# Operator verification for the current GSC benchmark path

This note records a repository-local sanity check of the exact operator and eigensolver branch used by the current GSC benchmark.

## Configuration checked

- Point cloud: connected Gaussian mixture with `N = 120`, `d = 2`
- Graph: `8`-nearest-neighbor directed graph from `kneighbors_graph`
- Measure: `teleporting_undirected_measure(W, alpha=0.5, t=5)`
- Implementation files:
  - `scikit-learn/sklearn/cluster/_spectral.py`
  - `scikit-learn/sklearn/manifold/_spectral_embedding.py`
  - `scikit-learn/sklearn/manifold/_laplacian.py`

## What was verified

- The benchmark configuration `laplacian_method="norm"` uses the normalized generalized Laplacian.
- On the checked Gaussian `k`-NN graph, the normalized generalized Laplacian is numerically symmetric:
  - `max(abs(L_norm - L_norm.T)) = 0.0`
- The random-walk generalized Laplacian is numerically non-symmetric:
  - `max(abs(L_rw - L_rw.T)) = 0.22053870521389596`
- Despite that non-symmetry, the random-walk operator had real eigenvalues up to numerical precision on this check:
  - `max(abs(imag(eigvals(L_rw)))) = 0.0`
- By instrumenting `_spectral_embedding`, the actual solver branch was:
  - `laplacian_method="norm"` -> `eigsh(..., sigma=1.0, which="LM")`
  - `laplacian_method="random_walk"` -> `eigs(..., sigma=1.0, which="LM")`

## Consequence for the manuscript

- The theoretical note should distinguish the general random-walk GSC operator, which is non-symmetric but self-adjoint in the weighted inner product, from the normalized GSC operator used in the benchmark figures.
- The current complexity analysis of the benchmark should therefore treat the implemented run as a symmetric shift-invert `eigsh` computation.
