# Graph Calinski-Harabasz Index via Random Walk Distance

## Overview

This document derives a **graph-adapted Calinski-Harabasz (CH) index** that measures
clustering quality on directed or undirected networks using a distance based on the
**transition matrix** (random walk matrix) P = D^{-1} A.

The key idea is:
1. Interpret P as defining a random walk on the graph.
2. Define a **diffusion distance** between nodes from the rows of P^k (or a polynomial filter thereof).
3. Replace Euclidean distances in the standard CH index with these graph distances.
4. Generalise to **discrete polynomial filters** g(P) = sum_k a_k P^k for multi-scale analysis.

---

## 1. Transition Matrix and Random Walk

Given adjacency matrix A (possibly directed, possibly weighted), define:

```
d_i = sum_j A_{ij}        (out-degree of node i)
P = D^{-1} A              (row-stochastic transition matrix)
```

where D = diag(d_1, ..., d_n). Entry P_{ij} is the probability of walking from i to j
in one step. Dangling nodes (d_i = 0) are handled by setting their row to 1/n (uniform
teleportation), ensuring P remains row-stochastic.

**Reference**: This is the standard random walk normalisation used throughout spectral
clustering literature (von Luxburg, 2007).

---

## 2. Diffusion Distance

The **diffusion distance** at time t between nodes i and j, introduced by
Coifman & Lafon (2006), measures how similarly the random walk behaves when started
from i vs j after t steps:

```
D_t(i, j)^2 = sum_z [ P^t(i,z) - P^t(j,z) ]^2 / phi(z)
```

where:
- P^t(i,z) is the (i,z)-entry of P^t: the probability of reaching z from i in t steps.
- phi(z) is a **reference measure** on nodes.

### Choice of Reference Measure phi

Several options exist:

| phi(z)        | Name              | Effect                                        |
|---------------|-------------------|-----------------------------------------------|
| 1             | Uniform           | All nodes weighted equally                    |
| pi(z)         | Stationary dist.  | Original Coifman-Lafon; down-weights hubs     |
| d_z / sum d   | Degree-based      | Approximation to stationary (for undirected)  |

For directed graphs where the stationary distribution may not exist or may be hard to
compute, the **uniform** weighting phi(z) = 1 is the safest default:

```
D_t(i, j)^2 = || e_i^T P^t - e_j^T P^t ||_2^2 = || row_i(P^t) - row_j(P^t) ||_2^2
```

This is simply the squared L2 distance between rows of P^t.

**Reference**: Coifman, R. R. & Lafon, S. (2006). "Diffusion maps." Applied and
Computational Harmonic Analysis, 21(1), 5-30.

---

## 3. Generalisation: Polynomial Graph Filters

Instead of using a single power P^t, we can use a **polynomial filter**:

```
g(P) = sum_{k=0}^{K} a_k P^k
```

where {a_k} are filter coefficients. The filtered representation of node i is row i of g(P),
and the filtered distance is:

```
d_g(i, j)^2 = || row_i(g(P)) - row_j(g(P)) ||_2^2
```

### Spectral Interpretation

If P has eigendecomposition P = V diag(mu_l) V^{-1}, then:

```
g(P) = V diag(g(mu_l)) V^{-1}
```

where g(mu) = sum_k a_k mu^k. The filter transforms each eigenvalue independently,
allowing targeted spectral shaping:
- **Low-pass** (keep large |mu|): emphasises cluster structure
- **Band-pass**: reveals structure at specific scales
- **Multi-scale average**: captures structure across scales

### Special Cases

| Filter               | Coefficients a_k          | Meaning                          |
|----------------------|---------------------------|----------------------------------|
| Single scale P^t     | a_t = 1, rest = 0         | Diffusion at time t              |
| Uniform average      | a_k = 1/(K+1) for k=0..K | Multi-scale mean                 |
| Exponential decay    | a_k = (1-r) r^k           | Geometric smoothing              |
| Custom weights       | user-defined               | Tailored spectral shaping        |

**Reference**: Defferrard, M., Bresson, X. & Vandergheynst, P. (2016). "Convolutional
Neural Networks on Graphs with Fast Localized Spectral Filtering." NeurIPS 2016.

**Reference**: Hammond, D. K., Vandergheynst, P. & Gribonval, R. (2011). "Wavelets on
Graphs via Spectral Graph Theory." ACHA, 30(2), 129-150.

---

## 4. Graph Calinski-Harabasz Index

### Standard CH (Euclidean)

The Calinski-Harabasz index (Calinski & Harabasz, 1974) is:

```
CH = [BCSS / (k-1)] / [WCSS / (n-k)]
```

where:
- WCSS = sum_c sum_{x in C_c} || x - mu_c ||^2   (within-cluster sum of squares)
- BCSS = sum_c n_c || mu_c - mu ||^2              (between-cluster sum of squares)
- k = number of clusters, n = number of points
- mu_c = centroid of cluster c, mu = global centroid

### Adapting to Graph Distances (No Centroids Needed)

On a graph there is no natural centroid. We use the **pairwise distance identity**:

```
sum_{x in C} || x - mu_C ||^2 = (1 / 2|C|) sum_{x,y in C} || x - y ||^2
```

This holds for any Euclidean-embeddable distance (which the L2 row distance of g(P) is).
So we define:

```
WCSS_G = sum_{c=1}^{k} (1 / 2n_c) sum_{x,y in C_c} d_g(x,y)^2
```

For the total sum of squares:

```
TSS_G = (1 / 2n) sum_{x,y} d_g(x,y)^2
```

And between-cluster:

```
BCSS_G = TSS_G - WCSS_G
```

Finally:

```
CH_G = [BCSS_G / (k-1)] / [WCSS_G / (n-k)]
```

### Efficient Computation

The within-cluster sum of pairwise squared distances can be computed efficiently.
Let Z = g(P) be the n x n filtered matrix. Then:

```
sum_{x,y in C_c} d_g(x,y)^2 = sum_{x,y in C_c} || Z_x - Z_y ||^2
                              = 2 n_c sum_{x in C_c} || Z_x ||^2 - 2 || sum_{x in C_c} Z_x ||^2
```

This avoids the O(n_c^2) pairwise computation, giving O(n_c * n) per cluster instead.

Similarly for the total:

```
sum_{x,y} d_g(x,y)^2 = 2n sum_x || Z_x ||^2 - 2 || sum_x Z_x ||^2
```

**Reference**: Calinski, T. & Harabasz, J. (1974). "A dendrite method for cluster analysis."
Communications in Statistics, 3(1), 1-27.

---

## 5. Weighted Variant (Stationary Distribution Weighting)

When the stationary distribution pi exists, one can use the weighted diffusion distance:

```
D_t(i,j)^2 = sum_z [ P^t(i,z) - P^t(j,z) ]^2 / pi(z)
```

This gives higher weight to transitions through low-probability nodes, making the distance
more sensitive to fine structure. In matrix form, define:

```
Z_weighted = g(P) @ diag(1 / sqrt(phi))
```

Then d_g(i,j)^2 = || Z_weighted[i,:] - Z_weighted[j,:] ||^2.

The CH index is computed identically but using Z_weighted instead of Z.

---

## 6. Implementation Notes

### Sparse Computation
- P^k is computed via repeated sparse matrix-vector or sparse matrix-matrix products.
- For the CH computation, we only need Z = g(P), which is at most K sparse mat-mat
  multiplications. If P is sparse (nnz << n^2), each multiplication is O(nnz * n).
- For very large graphs, one can compute Z row-by-row or in blocks.

### Handling Directed Graphs
- P = D^{-1} A is naturally asymmetric for directed graphs — no symmetrisation needed.
- The diffusion distance D_t(i,j) is symmetric even when P is not, because it compares
  rows of P^t (forward transition probabilities from i and j).

### Numerical Stability
- Dangling nodes (zero out-degree) get uniform transition: P_i = 1/n.
- Filter coefficients should be chosen so that g(1) > 0 (preserves stationary mode).
- Rows of Z are normalised implicitly by the stochasticity of P.

---

## 7. Summary of Key References

1. **Coifman & Lafon (2006)**. "Diffusion maps." ACHA 21(1), 5-30.
   - Introduced diffusion distance D_t via transition matrix powers.

2. **Calinski & Harabasz (1974)**. "A dendrite method for cluster analysis." Comm. Stat. 3(1), 1-27.
   - Original CH index (variance ratio criterion).

3. **Fouss, Pirotte, Renders & Saerens (2007)**. "Random-Walk Computation of Similarities
   between Nodes of a Graph." IEEE TKDE 19(3), 355-369.
   - Commute time, hitting time, resistance distance from random walks.

4. **Von Luxburg (2007)**. "A tutorial on spectral clustering." Statistics and Computing 17(4), 395-416.
   - Connections between random walks and spectral clustering.

5. **Defferrard, Bresson & Vandergheynst (2016)**. "Convolutional Neural Networks on Graphs
   with Fast Localized Spectral Filtering." NeurIPS 2016.
   - Polynomial (Chebyshev) graph filters.

6. **Hammond, Vandergheynst & Gribonval (2011)**. "Wavelets on Graphs via Spectral Graph Theory."
   ACHA 30(2), 129-150.
   - Spectral graph wavelets using polynomial filters.
