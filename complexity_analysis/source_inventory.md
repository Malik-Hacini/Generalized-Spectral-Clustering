# Source inventory for `complexity_analysis/`

This file records the main external and local sources used for the sparse-graph complexity analysis in `complexity_analysis/complexity.typ`.

## External sources

- Ulrike von Luxburg, *A Tutorial on Spectral Clustering* (2007)
  - URL: https://doi.org/10.1007/s11222-007-9033-z
  - Used for: sparse-vs-dense graph discussion, practical recommendation for `k`-nearest-neighbor graphs, and the role of sparse eigensolvers / eigengaps.

- Jianbo Shi and Jitendra Malik, *Normalized Cuts and Image Segmentation* (2000)
  - URL: https://doi.org/10.1109/34.868688
  - Used for: classical normalized spectral clustering baseline.

- Andrew Y. Ng, Michael I. Jordan, and Yair Weiss, *On Spectral Clustering: Analysis and an Algorithm* (2001)
  - URL: https://papers.nips.cc/paper/2092-on-spectral-clustering-analysis-and-an-algorithm
  - Used for: normalized spectral embedding / clustering baseline.

- scikit-learn user guide: Nearest Neighbors
  - URL: https://scikit-learn.org/stable/modules/neighbors.html
  - Used for: exact neighbor-search backends, heuristic complexity discussion, and `algorithm="auto"` behavior.

- scikit-learn API docs: SpectralClustering
  - URL: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
  - Used for: public pipeline description and default solver / labeling options.

- scikit-learn user guide: clustering / spectral clustering section
  - URL: https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering
  - Used for: graph-partitioning interpretation and label-assignment remarks.

- scikit-learn API docs: KMeans
  - URL: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
  - Used for: practical k-means complexity and restart behavior.

- SciPy ARPACK tutorial
  - URL: https://docs.scipy.org/doc/scipy/tutorial/arpack.html
  - Used for: shift-invert explanation and why smallest eigenvalues are treated through a transformed problem.

- SciPy `eigsh` documentation
  - URL: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
  - Used for: actual solver interface used by the repository, the `ncv` recommendation, and the internal sparse LU / iterative solve statement when `sigma` is set.

- SciPy `eigs` documentation
  - URL: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html
  - Used for: nonsymmetric fallback remarks.

- R. B. Lehoucq, D. C. Sorensen, and C. Yang, *ARPACK Users' Guide* (1998)
  - URL: https://epubs.siam.org/doi/book/10.1137/1.9780898719628
  - Used for: IRAM / IRLM storage model and the tradeoff between matrix-vector products, restart cost, and orthogonality maintenance.

- Jon L. Bentley, *Multidimensional Binary Search Trees Used for Associative Searching* (1975)
  - URL: https://doi.org/10.1145/361002.361007
  - Used as the original KD-tree reference behind the scikit-learn nearest-neighbor discussion.

## Local sources

- `latex/mainv7.tex`
  - Used for notation and mathematical conventions.

- `complexity_benchmark_clean.py`
  - Used to match the exact benchmarked SC / GSC configuration.

- `scikit-learn/sklearn/cluster/_spectral.py`
  - Used to confirm graph construction, symmetrization, measure resolution, and label assignment.

- `scikit-learn/sklearn/manifold/_spectral_embedding.py`
  - Used to confirm connectivity checks and the default ARPACK shift-invert path.

- `scikit-learn/sklearn/manifold/_laplacian.py`
  - Used to confirm the exact standard and generalized Laplacian constructions.

- `scikit-learn/sklearn/neighbors/_graph.py`, `scikit-learn/sklearn/neighbors/_unsupervised.py`, and `scikit-learn/sklearn/neighbors/_base.py`
  - Used to confirm how `kneighbors_graph` delegates to exact nearest-neighbor backends and how `algorithm="auto"` is chosen.

- `competitors/measures.py`
  - Used to confirm that the benchmarked GSC measure uses repeated sparse vector-times-matrix multiplication rather than explicit matrix powering.

- `complexity_analysis/operator_verification.md`
  - Used to record the local Gaussian `k`-NN sanity check distinguishing the symmetric normalized GSC operator from the non-symmetric random-walk operator.
