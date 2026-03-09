
= Sparse-Graph Complexity of Standard SC and GSC

This note analyzes the complexity of standard spectral clustering (SC) and generalized spectral clustering (GSC) in the sparse-graph regime targeted by this repository. The emphasis is deliberately narrow: we analyze the actual code path used by `complexity_benchmark_clean.py`, then separate that implementation-sensitive discussion from the more classical Krylov-style asymptotic model. This distinction is essential, because many textbook statements about spectral clustering complexity tacitly assume a sparse-matrix Lanczos iteration, while the scikit-learn `arpack` path used here relies on shift-invert mode and therefore solves sparse linear systems internally rather than only applying sparse matrix-vector products @luxburg2007 @scipy-arpack-tutorial @scipy-eigsh.

== Scope and notation

For this section only, we reserve $k$ for the sparsity parameter of the graph. This keeps the notation compatible with $k$-nearest-neighbor graphs. To avoid a collision with the usual spectral-clustering notation, we denote by $r$ the number of requested clusters and eigenvectors.

The standing symbols are:

- $N$: number of samples or graph vertices.
- $d$: ambient data dimension for point-cloud input.
- $k$: sparsity parameter. Let $m$ denote the number of stored nonzero entries of $W$. For a directed $k$-nearest-neighbor graph each row has exactly $k$ outgoing edges, so $m = N k$. For a general sparse network we assume $m = Theta(N k)$.
- $r$: number of clusters and number of requested spectral coordinates.
- $t$: diffusion depth used by the GSC measure.
- $I$: number of Lloyd iterations in a single `kmeans` run.
- $s$: number of `kmeans` restarts.
- $p$: ARPACK Krylov subspace size (`ncv` in SciPy), with the SciPy recommendation $p > 2 r$ @scipy-eigsh.
- $q$: number of Krylov updates or restarted outer iterations until convergence.

We consider two input regimes.

- *Point-cloud input.* We start from $X in RR^(N times d)$, build a sparse graph, and then run SC or GSC on that graph.
- *Precomputed sparse network input.* We start from a sparse adjacency matrix $W in RR^(N times N)$ and skip graph construction.

The repository benchmark uses the normalized Laplacian path (`laplacian_method="norm"`) together with `assign_labels="kmeans"`. The unnormalized branch in the vendored implementation converts the sparse Laplacian to dense and calls `eigh`, so it is outside the sparse large-scale regime analyzed here.

== Standard SC in the scikit-learn pipeline

=== What the code actually does

For the benchmarked standard baseline, `complexity_benchmark_clean.py` constructs

`SpectralClustering(affinity="nearest_neighbors" or "precomputed", standard=True, laplacian_method="norm", assign_labels="kmeans")`.

After validation, the vendored `SpectralClustering.fit` routine does the following:

1. if the input is a point cloud, build a directed $k$-nearest-neighbor connectivity graph with `kneighbors_graph`;
2. symmetrize it as $W_s = 1/2 (W + W^T)$ because `standard=True`;
3. call the spectral embedding routine on the normalized Laplacian;
4. run `kmeans` on the $N times r$ embedding.

The implementation also performs a connectivity check before the eigensolver, and its Laplacian helper computes the row-normalized matrix $P = D^(-1) W$ even in `standard=True` mode. That extra normalization pass does not change the asymptotic order, but it does add a real linear-time constant.

=== Stage 1: exact $k$-NN graph construction

The graph-construction step is the most delicate part of the sparse point-cloud analysis. The scikit-learn documentation is explicit that the exact nearest-neighbor backend can be `brute`, `kd_tree`, or `ball_tree`, and that `algorithm="auto"` chooses among them using heuristics tied to the data dimension, the requested neighborhood size, sparsity of the input representation, and metric support @sklearn-neighbors. The local vendored code confirms the same rule.

More precisely, `algorithm="auto"` falls back to `brute` when the input is sparse, when the metric is precomputed, when $d > 15$, when $k >= N/2$, or when the metric is unsupported by both tree backends. Therefore, one cannot quote a single exact asymptotic formula for `kneighbors_graph` without first fixing the backend regime.

*Brute-force regime.* The scikit-learn nearest-neighbor guide states that the naive all-pairs search scales as $O(d N^2)$ @sklearn-neighbors. This is the correct time order for exact brute-force $k$-NN graph construction. The current implementation does _not_ need to materialize the full dense $N times N$ distance matrix: it computes nearest neighbors through pairwise-distance reductions and returns only the sparse graph. Thus the dominant time is quadratic in $N$, but the persistent output memory remains $Theta(N k)$ rather than $Theta(N^2)$.

*Tree-based regime.* In low dimension, scikit-learn documents approximate query costs of order $O(d log N)$ for KD-trees and approximately $O(d log N)$ for Ball trees on favorable data, together with the caveat that KD-tree performance degrades toward brute force in high dimension, and that Ball-tree performance depends strongly on data geometry @sklearn-neighbors. Because the full graph requires one neighbor query per sample, a safe decomposition is

$
G_t = B(N, d) + N Q(N, d, k) + O(N k).
$

Under the usual favorable low-dimensional exact-search assumptions, one may summarize this as

$
G_t = O(N log N) + O(N (d log N + k)).
$

Here $B$ is the tree-build cost and $Q$ is the per-query cost. This should be read as an optimistic sparse-geometry model, not as a dimension-free worst-case theorem. In the exact worst case, tree queries again degrade to quadratic behavior.

*Resulting point-cloud graph-construction bound.* If we write $G(N, d, k)$ for the backend-dependent exact $k$-NN graph build, then the standard SC point-cloud pipeline starts with

$
G = G(N, d, k),
$

with

- $G = O(d N^2)$ in the brute regime;
- $G = O(N log N + N (d log N + k))$ in the favorable exact tree regime.

=== How dimension changes the dominant cost in practice

For point-cloud input, the role of the ambient dimension $d$ is easy to underestimate because the final graph is sparse while the neighbor search is performed in the original feature space. The correct practical picture is the following.

- *Low ambient dimension and favorable geometry.* If $d$ is small and the data have enough geometric structure for exact tree pruning to be effective, then KD-tree or Ball-tree search can reduce the graph-build cost to the quasi-linear regime described above. In that case, with $k = Theta(log N)$, the graph-construction stage is roughly $O(N d log N)$ up to lower-order $O(N log N)$ terms.
- *Moderate or high ambient dimension.* The scikit-learn heuristic switches `algorithm="auto"` to `brute` as soon as $d > 15$ @sklearn-neighbors, and even before that threshold exact tree search can deteriorate when pruning becomes ineffective. In this regime the graph build is $O(d N^2)$, so the factor $N^2$ is dominant in $N$ while $d$ multiplies the entire quadratic cost.
- *Large neighborhood size.* If $k >= N/2$, the same heuristic also switches to `brute` @sklearn-neighbors. Thus increasing the sparsity parameter too aggressively destroys the tree advantage even if $d$ is small.

This yields a clean rule of thumb for the benchmarked point-cloud pipeline.

- If `auto` selects `brute`, the neighbor-search stage is usually the dominant term of the whole pipeline, because $O(d N^2)$ overwhelms the later linear sparse passes.
- If `auto` selects a tree backend and pruning is effective, the graph build becomes much cheaper and the dominant term can move to the spectral stage, in particular to the shift-invert factorization/solve cost.
- The output sparsity $m = Theta(N k)$ only controls the cost _after_ the graph exists. It does not by itself determine the cost of discovering the neighbors from $X$.

The distinction between ambient dimension and intrinsic dimension is also important here. The scikit-learn user guide stresses that tree performance depends strongly on the structure of the data and can be much better when the data occupy a lower-dimensional manifold inside the ambient space @sklearn-neighbors. However, the implementation heuristic only sees the ambient feature count $d$, so in practice the automatic routine is conservative: once $d$ is moderately large, it chooses brute force even if the data may have lower intrinsic dimension.

=== Stage 2: symmetrization

Once the sparse directed connectivity graph is built, standard SC replaces it with

$
W_s = 1/2 (W + W^T).
$

If $W$ has $m$ nonzeros, then $W^T$ has the same number of nonzeros and the sparsity pattern of $W_s$ is contained in the union of the supports of $W$ and $W^T$. Hence

$W_s$ has at most $2 m$ stored off-diagonal entries.

Therefore transpose, sparse addition, and scalar rescaling all cost $O(m)$ time and $O(m)$ memory.

*Proof.* A sparse transpose only reindexes the existing nonzero entries. Sparse addition touches each stored entry once. The factor $1/2$ is a scalar multiplication on the stored values. No operation creates more than one output nonzero per nonzero in $W$ or $W^T$. Therefore the total time is linear in the number of stored entries, and the final number of stored entries is at most $2 m$. #parbreak()

=== Stage 3: normalized Laplacian assembly

For the normalized standard pipeline, the helper class first computes the degree vector, then constructs the standard Laplacian, then rescales by the square roots of the degrees. Every substep is linear in the number of stored edges.

*Proposition 1.* If the symmetrized sparse affinity matrix has $m_s$ nonzeros, then the normalized standard Laplacian used in the benchmark can be assembled in $O(m_s + N)$ time and stored in $O(m_s + N)$ memory.

*Proof.* Degree extraction is one sparse row-sum pass, hence $O(m_s)$. The sparse unnormalized Laplacian is obtained from a sparse diagonal and the sparse adjacency, so it is also $O(m_s + N)$. The final normalization is an entrywise scaling of the stored nonzeros and therefore costs another $O(m_s)$. Since $m_s = Theta(m)$ in the sparse regime, the total order is linear in $m$. #parbreak()

The connectivity check and the symmetry check in the embedding routine are again linear passes over the graph. They do not change the order, but they are real implementation costs and should be measured in component-wise timings later.

=== Stage 4: eigensolver cost

This stage is where one must distinguish the abstract sparse-Lanczos model from the actual scikit-learn implementation.

*Classical sparse-Krylov model.* If one solves for the first $r$ eigenvectors of a sparse symmetric Laplacian with a standard Lanczos or implicitly restarted Lanczos procedure that accesses the matrix only through sparse matrix-vector products, then one Krylov update costs one sparse matvec plus orthogonalization against the current basis. ARPACK explicitly states that its performance trades off the cost of the user-supplied matrix-vector products, the restart mechanism, and the maintenance of Lanczos orthogonality; it also gives storage of order $N O(p) + O(p^2)$ for a $p$-dimensional Krylov basis @arpack1998. In that idealized model,

$
E_0 = O(q m) + O(q N p),
$

and, with the standard choice $p = O(r)$,

$
E_0 = O(q m) + O(q N r).
$

For fixed $r$, this is linear in the graph size up to the convergence factor $q$, which itself depends on spectral separation; von Luxburg emphasizes that sparse eigensolvers accelerate when the relevant eigengap is larger @luxburg2007.

*Actual scikit-learn `arpack` path.* The vendored implementation does _not_ call `eigsh` on the smallest eigenvalues in standard mode. Instead it negates the normalized Laplacian and calls

`eigsh(laplacian, k=r, sigma=1.0, which="LM")`,

that is, ARPACK in shift-invert mode. The SciPy documentation states that when `sigma` is specified, the routine internally solves linear systems with the shifted operator and, for explicit sparse matrices, does so through a sparse LU decomposition unless the user supplies a custom inverse operator @scipy-arpack-tutorial @scipy-eigsh. Therefore the faithful implementation-level cost is

$
E_1 = F(N, m) + q U(N, m) + O(q N p).
$

Here $F$ is the cost of factoring the shifted sparse operator and $U$ is the cost of one solve with the factors. A crucial caution follows.

*Caution.* One cannot, in general, compress $F$ and $U$ into a pure $O(m)$ expression that depends only on the stored-edge count. Sparse factorization cost depends on fill-in, ordering, and graph geometry, not just on the original number of nonzeros. Consequently, the commonly quoted $O(q m)$ sparse-Lanczos estimate is a good theoretical baseline for spectral clustering, but it is _not_ the exact complexity law of the default scikit-learn `arpack` path.

This distinction matters for both SC and GSC.

=== Stage 5: label assignment

The benchmark uses `assign_labels="kmeans"`. The scikit-learn documentation states the practical average complexity as $O(k n T)$ for $k$ clusters, $n$ samples, and $T$ Lloyd iterations @sklearn-kmeans. In the spectral-clustering setting the embedding has dimension $r$, and the number of centers is also $r$, so a direct per-iteration count gives $O(N r^2)$ work for the assignment step plus centroid updates of lower order. Therefore the total labeling cost is

$
L = O(s I N r^2).
$

When $r$ is fixed, this is linear in $N$ and typically lower order than the eigensolver or the brute-force graph build.

=== Standard SC: resulting sparse-graph complexity

If the sparse graph is given as input, the benchmarked standard SC pipeline satisfies

$
T_S^(n) = O(m) + E_1 + O(s I N r^2),
$

where the $O(m)$ term collects connectivity checking, optional symmetrization, degree extraction, normalization, diagonal updates, and other linear sparse passes.

If the input is a point cloud,

$
T_S^(p) = G(N, d, k) + O(m) + E_1 + O(s I N r^2).
$

Under the abstract sparse-Krylov model, one may replace $E_1$ by $O(q m + q N r)$, but that replacement is a mathematical idealization rather than the exact scikit-learn default.

== GSC in the same sparse-graph regime

The GSC pipeline differs from standard SC in exactly the places that matter for complexity:

- it does not symmetrize the adjacency before preprocessing;
- it computes a vertex measure $nu$;
- it computes $xi = P^T nu$;
- it assembles a generalized Laplacian rather than the standard one.

The benchmarked GSC measure is the callable `teleporting_undirected_measure`, resolved _before_ the Laplacian helper is created. This detail is important: the benchmark therefore uses repeated sparse vector-times-matrix multiplication rather than explicit matrix powering.

=== Stage 1: graph construction

For point-cloud input, GSC pays exactly the same graph-construction cost as standard SC because both pipelines call the same `kneighbors_graph` routine. Thus

$
G = G(N, d, k).
$

For precomputed network input this stage is absent.

=== Stage 2: measure computation

The benchmarked GSC measure is

$
nu = ((1 / N) 1^T P^t)^alpha,
$

implemented through the recurrence

$
v_(0) = (1 / N) 1^T, v_(s+1) = v_s P, nu = v_t^alpha.
$

followed by clipping and normalization.

*Proposition 2.* Let $W$ be sparse with $m$ stored nonzero entries. Then the callable GSC measure used in the benchmark can be computed in $O(m + t m + N)$ time and $O(m + N)$ memory.

*Proof.* Building the degree vector and the row-stochastic matrix $P = D^(-1) W$ requires one sparse row-sum pass and one sparse scaling pass, hence $O(m)$. Because $P$ has the same sparsity pattern as $W$, one update $v -> v P$ touches each stored edge once and costs $O(m)$. Repeating this update $t$ times gives $O(t m)$. The elementwise power, positivity correction, and normalization are all $O(N)$. The sparse matrix $P$ and a constant number of dense vectors are stored, giving $O(m + N)$ memory. #parbreak()

In the sparse $k$-graph regime $m = Theta(N k)$, so for fixed $t$ the measure stage is linear in the graph size.

=== Stage 3: computation of $xi$

After $nu$ is available, the implementation computes

$
xi = P^T nu.
$

This is one sparse matrix-vector product and therefore costs $O(m)$ time and $O(N)$ extra memory.

=== Stage 4: generalized Laplacian assembly

The normalized GSC path is based on

$
L_nu = D_(nu + xi) - (D_nu P + P^T D_nu),
$

followed by the normalization

$
L_nu^n = D_(nu + xi)^(-1/2) L_nu D_(nu + xi)^(-1/2).
$

Two structural facts are immediate.

*Proposition 3.* If $P$ has $m$ nonzeros, then $D_nu P$ has exactly the same sparsity pattern as $P$, $P^T D_nu$ has the same sparsity pattern as $P^T$, and therefore $L_nu$ has at most $2 m + N$ stored entries.

*Proof.* Left or right multiplication by a diagonal matrix rescales existing nonzeros but cannot create new off-diagonal positions. Thus $D_nu P$ has the same nonzero pattern as $P$, and $P^T D_nu$ has the same nonzero pattern as $P^T$. Adding the diagonal contributes at most $N$ additional stored positions. #parbreak()

*Proposition 4.* The normalized generalized Laplacian can be assembled in $O(m + N)$ time and stored in $O(m + N)$ memory.

*Proof.* By Proposition 3, all sparse products and sparse sums act on at most $2 m + N$ stored entries. The final normalization is again an entrywise scaling of the stored entries. Therefore the total order is linear in $m + N$. #parbreak()

An additional and very important consequence is symmetry.

*Proposition 5.* The random-walk generalized Laplacian

$
L_("rw", nu) = D_(nu + xi)^(-1) L_nu
$

is generally not symmetric in the Euclidean inner product, but it is self-adjoint in the weighted inner product of $ell^2(cal(V), nu + xi)$. Consequently its spectrum is real.

*Proof.* From the construction above, $L_nu$ is symmetric. Therefore for any vectors $f$ and $g$,

$
< f, L_("rw", nu) g >_(nu + xi)
  = f^T D_(nu + xi) D_(nu + xi)^(-1) L_nu g
  = f^T L_nu g.
$

Since $L_nu$ is symmetric, $f^T L_nu g = (L_nu f)^T g$. Reversing the same calculation gives

$
(L_nu f)^T g = < L_("rw", nu) f, g >_(nu + xi).
$

Hence $< f, L_("rw", nu) g >_(nu + xi) = < L_("rw", nu) f, g >_(nu + xi)$, so $L_("rw", nu)$ is self-adjoint in the weighted space. A self-adjoint operator has real eigenvalues. #parbreak()

The normalized generalized Laplacian used by the current benchmark is different:

$
L_nu^n = D_(nu + xi)^(-1/2) L_nu D_(nu + xi)^(-1/2).
$

Because it is obtained from the symmetric matrix $L_nu$ by diagonal conjugation, it is itself symmetric.

=== What the current run actually does

The benchmark in `complexity_benchmark_clean.py` sets `laplacian_method="norm"`, not `laplacian_method="random_walk"`. Therefore the actual operator used in the current complexity figures is the normalized generalized Laplacian $L_nu^n$, not the random-walk operator $L_("rw", nu)$.

A repository-local sanity check on a connected Gaussian $8$-NN graph with $N = 120$ and $d = 2$ confirms the distinction recorded in `complexity_analysis/operator_verification.md`.

- For the normalized GSC operator, `max(abs(L_norm - L_norm.T)) = 0.0`.
- For the random-walk GSC operator, `max(abs(L_rw - L_rw.T)) = 0.22053870521389596` while the computed eigenvalues were real up to numerical precision.
- By instrumenting `_spectral_embedding`, the normalized benchmark path called `eigsh(..., sigma=1.0, which="LM")`, whereas the random-walk path called `eigs(..., sigma=1.0, which="LM")`.

This distinction is important for the complexity analysis. The paper-level statement about self-adjoint but non-symmetric GSC applies to the random-walk operator, while the current benchmarked implementation path uses the symmetric normalized operator.

=== Stage 5: eigensolver and label assignment

Because the normalized GSC Laplacian is symmetric, the benchmarked GSC pipeline uses the same shift-invert `eigsh` path and the same `kmeans` labeling step as standard SC. If one were instead to benchmark the random-walk operator, the code would move to the nonsymmetric `eigs` path. For the current benchmark, the solver and labeling costs are

$
E_G = E_1, L_G = O(s I N r^2).
$

=== GSC: resulting sparse-graph complexity

For precomputed sparse network input,

$
T_G^(n) = O(t m) + O(m) + E_1 + O(s I N r^2).
$

The first $O(t m)$ term is the measure computation, while the second $O(m)$ term collects the $xi$ computation, connectivity checks, Laplacian assembly, and other linear sparse passes.

For point-cloud input,

$
T_G^(p) = G(N, d, k) + O(t m) + O(m) + E_1 + O(s I N r^2).
$

If $t$ is treated as a fixed hyperparameter, then GSC adds only linear-in-$m$ preprocessing on top of the standard SC sparse graph build and sparse spectral stage.

== Direct SC versus GSC comparison

=== Precomputed sparse network input

If the sparse graph is already given, both methods have the same global structure:

- one or a few linear sparse preprocessing passes;
- one sparse spectral solve for $r$ coordinates;
- one clustering step on the $N times r$ embedding.

The difference is in the preprocessing constants.

- Standard SC pays for symmetrization and standard Laplacian assembly: $O(m)$.
- GSC pays for measure computation, $xi$, and generalized Laplacian assembly: $O(t m) + O(m)$.

Therefore, for fixed $t$, SC and GSC have the same sparse-order preprocessing complexity, namely linear in the number of stored edges. GSC is more expensive by a constant-factor number of additional sparse passes, but not by a larger asymptotic order.

If one adopts the abstract sparse-Krylov model for the eigensolver, then both methods satisfy

$
T^(n) = O(m) + O(q m + q N r) + O(s I N r^2),
$

up to the extra $t m$ factor in GSC. In contrast, under the exact default scikit-learn implementation the dominant term can instead be the shift-invert factorization,

$
T^(n) = O(m) + F(N, m) + q U(N, m) + O(q N p) + O(s I N r^2),
$

for both SC and GSC.

=== Point-cloud input

For point clouds, the graph-construction stage is often the decisive difference between sparse and dense complexity claims.

- In the brute regime, both SC and GSC inherit an exact $k$-NN build cost of $O(d N^2)$. This stage dominates all linear sparse passes and often dominates the entire pipeline.
- In the favorable exact tree regime, both methods reduce to near-linear or quasi-linear graph construction in $N$ for fixed dimension and mild data geometry assumptions.
- After the graph is built, SC adds only symmetrization, while GSC adds measure computation and generalized Laplacian assembly. These are all linear in $m = Theta(N k)$.

Hence, for fixed $t$, the asymptotic comparison on point clouds is

$
T_S^(p) = G(N, d, k) + O(m) + E_1 + O(s I N r^2),
$

$
T_G^(p) = G(N, d, k) + O(t m) + O(m) + E_1 + O(s I N r^2).
$

Thus GSC does _not_ alter the order of the expensive point-cloud graph build. It only adds linear sparse preprocessing after the graph exists.

=== Specialization to $k = Theta(log N)$

The main paper uses a logarithmic nearest-neighbor scale. In that regime,

$
m = Theta(N log N).
$

Therefore all sparse linear passes in either SC or GSC become $Theta(N log N)$.

If, additionally, the exact $k$-NN graph can be built in the favorable tree regime and $d$, $r$, and $t$ are treated as constants, then the non-eigensolver part of both pipelines is quasi-linear in $N$.

If instead the auto-backend falls back to brute force, then the same logarithmic sparsity of the _output graph_ does not prevent a quadratic graph-construction time. This is why it is misleading to infer point-cloud complexity only from the final graph sparsity. The cost of obtaining the sparse graph can still be $Theta(d N^2)$.

=== Clean theorem-like summary for $k = Theta(log N)$

#block(stroke: 0.8pt + black, inset: 10pt, radius: 4pt)[
*Theorem (current sparse-logarithmic regime, single run).* Assume $k = Theta(log N)$, hence $m = Theta(N log N)$. Let $r$ be the number of requested clusters/eigenvectors, let $s$ be the number of `kmeans` restarts, let $I$ be the number of Lloyd iterations per restart, let $p > 2 r$ be the ARPACK subspace size, and let

$
E_1 = F(N, m) + q U(N, m) + O(q N p)
$

denote the implementation-level shift-invert eigensolver cost of the current scikit-learn `arpack` path.

Then the current benchmarked single-run complexities are:

- *Precomputed sparse network input.*
  - Standard SC:
    $
    T_S^(n) = O(N log N) + E_1 + O(s I N r^2).
    $
  - GSC:
    $
    T_G^(n) = O(t N log N) + E_1 + O(s I N r^2).
    $

- *Point-cloud input with exact $k$-NN graph construction.*
  - Standard SC:
    $
    T_S^(p) = G(N, d, Theta(log N)) + O(N log N) + E_1 + O(s I N r^2).
    $
  - GSC:
    $
    T_G^(p) = G(N, d, Theta(log N)) + O(t N log N) + E_1 + O(s I N r^2).
    $

Here the graph-build term satisfies two practically important regimes:

- *Brute-force regime.*
  $
  G(N, d, Theta(log N)) = O(d N^2).
  $
- *Favorable exact tree regime.*
  $
  G(N, d, Theta(log N)) = O(N log N + N(d log N + log N)) = O(N d log N).
  $

Consequently:

- if the auto-backend selects `brute`, the point-cloud $k$-NN stage is typically the dominant term of the whole run;
- if the auto-backend selects a tree backend and pruning remains effective, the graph build becomes quasi-linear and the dominant term can move to the spectral stage, namely the shift-invert factorization/solve cost $E_1$;
- for fixed $t$, GSC has the same sparse order as SC after the graph is available, but with an additional $O(t N log N)$ preprocessing term.
]

Under the more abstract sparse-Krylov model, one may replace $E_1$ by $O(q m + q N r)$, but this is a mathematical idealization rather than the exact cost of the default shift-invert implementation.

=== Naive grid search over $(t, alpha)$

Suppose the current GSC model selection evaluates $n_t$ values of $t$ and $n_alpha$ values of $alpha$, and suppose each parameter pair reruns the full `fit` pipeline from scratch exactly as the current code does. Then the grid size is

$
g = n_t n_alpha,
$

and the total cost of the current naive exhaustive search is exactly multiplied by $g$.

- *Network input.*
  $
  T_g^(n) = g T_G^(n).
  $
- *Point-cloud input.*
  $
  T_g^(p) = g T_G^(p).
  $

At present this multiplication applies to the whole GSC pipeline, including the eigensolver and, for point clouds, the repeated $k$-NN graph construction. Later optimizations may allow shared computation across grid points, but the current analysis should reflect the actual exhaustive rerun strategy.

== What can be claimed rigorously

The preceding derivations support the following statements.

*Claim 1.* On a fixed sparse graph with $m = Theta(N k)$ and fixed diffusion depth $t$, standard SC and GSC have the same preprocessing order: linear in $m$. GSC has a larger constant because it computes $nu$ and $xi$.

*Claim 2.* On point clouds, the dominant term can come from exact $k$-NN graph construction rather than from the spectral stage. In the brute regime this cost is quadratic in $N$ even though the final graph has only $Theta(N k)$ stored edges.

*Claim 3.* The general random-walk GSC operator is non-symmetric in the Euclidean inner product but self-adjoint in the weighted inner product, hence it has real eigenvalues. The current benchmark, however, uses the normalized operator, which is symmetric and therefore goes through the `eigsh` branch.

*Claim 4.* The often-quoted $O(q m)$ sparse-eigensolver law is an informative mathematical baseline, but it is not the exact complexity of the default scikit-learn `arpack` implementation because that implementation invokes shift-invert mode and therefore pays for sparse linear solves and factorization.

*Claim 5.* In the current naive exhaustive search over $(t, alpha)$, the total GSC cost is exactly multiplied by the grid size because the full pipeline is rerun for every grid point.

The following stronger claims would _not_ be justified without additional assumptions and should therefore be avoided.

- It is not justified to claim a universal $O(N k)$ eigensolver complexity for the default scikit-learn pipeline.
- It is not justified to claim that KD-tree or Ball-tree construction always yields the same asymptotic behavior as the final sparse graph size; high-dimensional degradation invalidates that shortcut.
- It is not justified to claim that GSC and SC have identical runtimes. They have the same sparse order under fixed $t$, but GSC performs extra linear sparse passes.

== Implementation-sensitive remarks for later experiments

The theoretical decomposition above suggests the exact component timings that should be isolated next.

- `kneighbors_graph` build time, separated by backend regime when possible.
- SC symmetrization time.
- GSC measure time, with $t$ varied explicitly.
- GSC $xi = P^T nu$ time.
- Standard and generalized Laplacian assembly time.
- Eigensolver wall time, together with whether the run uses `eigsh` or `eigs`.
- `kmeans` time on the spectral embedding.

Two repository-specific points are especially worth verifying experimentally.

- The efficient benchmarked GSC path is the callable measure in `competitors/measures.py`; the tuple-based `matrix_power` path inside `Laplacian` is a different and potentially much more expensive code path.
- The normalized sparse pipeline is the right object for large-scale comparison. The unnormalized branch densifies the matrix and should be analyzed separately if needed.

== References

#bibliography("references.bib", style: "ieee")
