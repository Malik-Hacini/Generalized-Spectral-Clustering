= Complexity comparison : SC and GSC

== Notation

- $N$: number of vertices or samples.
- $d$: ambient dimension of the point cloud.
- $k$: sparsity parameter. For a directed $k$-NN graph, the sparse graph has $m = N k$ stored edges.
- $m$: number of stored edges of the sparse graph.
- $r$: number of requested clusters and eigenvectors.
- $t$: GSC diffusion depth parameter.
- $s$: number of `kmeans` restarts.
- $I$: number of Lloyd iterations per restart.
- $q$: number of eigensolver iterations / restarted Krylov updates.


We compare the time complexity of a single run of the GSC algorithm and the classical SC algorithm. We consider a given sparse graph $G$ with $m$ stored edges. The additional GSC-specific operations are the computation of the measure $nu_(t,alpha)$ and the assembly of the generalized Laplacian $L_(nu_(t,alpha))$ or it's normalized counterpart $macron(L)_(nu_(t,alpha))$.

The measure  
- *Brute-force regime.*
  $
  G(N, d, k) = O(d N^2).
  $
- *Favorable exact tree regime.*
  $
  G(N, d, k) = O(N d log N).
  $

Let $E(m, r)$ denote the sparse spectral step, i.e. the cost of computing the first $r$ spectral coordinates from a sparse Laplacian.

- *Standard literature baseline.*
  $
  E(m, r) = O(q m + q N r).
  $
  For fixed $r$, this becomes $E(m, r) = O(q m)$.
- *Implementation caveat.* In the current scikit-learn benchmark path, the actual solver uses shift-invert ARPACK, so real wall-clock time can be larger and more variable than the baseline above. For reviewer-facing asymptotic comparison, however, the sparse-Krylov estimate is a reasonable standard model as long as it is used for both SC and GSC and explicitly identified as the algorithmic baseline rather than the exact implementation cost.

Throughout this summary, we specialize to

$
k = Theta(log N),
$

so that

$
m = Theta(N log N).
$

== Final results

=== Complexity of standard SC

#block(stroke: 0.8pt + black, inset: 10pt, radius: 4pt)[
*Result 1 (single-run standard SC).* Under $k = Theta(log N)$, the single-run complexity of standard spectral clustering is:

- *Sparse network input.*
  $
  T_S^n = O(N log N) + E(Theta(N log N), r) + O(s I N r^2).
  $

- *Point-cloud input.*
  $
  T_S^p = G(N, d, Theta(log N)) + O(N log N) + E(Theta(N log N), r) + O(s I N r^2).
  $

In particular:

- *Brute-force graph construction.*
  $
  T_S^p = O(d N^2) + E(Theta(N log N), r) + O(N log N).
  $
- *Favorable exact tree regime.*
  $
  T_S^p = O(N d log N) + E(Theta(N log N), r) + O(N log N).
  $
]

If one uses the standard sparse-Lanczos baseline with fixed $r$,

$
E(Theta(N log N), r) = O(q N log N).
$

Therefore:

- in the brute-force regime, the graph build usually dominates because $O(d N^2)$ is much larger than the sparse terms;
- in the favorable tree regime, the dominant term is not automatically the spectral term: it is the larger of $O(N d log N)$ and $O(q N log N)$. Thus the comparison is essentially controlled by `max(d, q)` up to constants.

=== Differences between GSC and SC for a single run

#block(stroke: 0.8pt + black, inset: 10pt, radius: 4pt)[
*Result 2 (single-run GSC overhead relative to SC).* Relative to standard SC, a single GSC run changes only the sparse preprocessing stage.

The additional GSC-specific operations are:

- computation of the measure $nu$: $O(t m)$,
- computation of $xi = P^T nu$: $O(m)$,
- assembly of the generalized Laplacian: $O(m)$.

Hence the extra single-run GSC overhead is

$
Delta_G = O(t m) + O(m).
$

Under $k = Theta(log N)$ this becomes

$
Delta_G = O(t N log N) + O(N log N).
$

So, for fixed $t$, the GSC-over-SC overhead is linear in the sparse graph size.
]

In the current benchmark path, both methods use the normalized Laplacian route and therefore the same symmetric sparse eigensolver branch. Consequently, the spectral term $E(m, r)$ and the label-assignment term $O(s I N r^2)$ are the same for SC and GSC.

=== Final conclusion: same asymptotics

#block(stroke: 1pt + black, inset: 10pt, radius: 4pt)[
*Result 3 (final asymptotic comparison).* For fixed $t$, SC and GSC have the same single-run asymptotic complexity.
   

- *Sparse network input.* Both methods have
  $
  O(N log N) + E(Theta(N log N), r) + O(s I N r^2)
  $
  single-run complexity, with GSC carrying a larger linear preprocessing constant.

- *Point-cloud input.* Both methods have
  $
  G(N, d, Theta(log N)) + E(Theta(N log N), r) + O(N log N) + O(s I N r^2)
  $
  single-run complexity, again with GSC carrying a larger linear sparse-preprocessing constant.

Therefore the asymptotic orders are the same for SC and GSC; the difference is a linear-in-$m$ GSC overhead, not a different asymptotic class.
]
