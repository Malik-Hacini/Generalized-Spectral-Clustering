# Agents Guide: `complexity_analysis/`

Read the repository-level `agents.md` first. This file adds folder-specific instructions for the complexity-analysis writing task.

## Mission

This folder is for a research-paper-quality complexity study of Generalized Spectral Clustering (GSC) versus standard Spectral Clustering (SC).

The main deliverable is a rigorous Typst manuscript in `complexity_analysis/complexity.typ` that does all of the following:

- gives a mathematically justified complexity analysis of GSC and SC,
- separates asymptotic-order claims from constant-factor overhead claims,
- distinguishes point-cloud input from precomputed-network input,
- uses the existing figures/results here as baseline evidence,
- adds component-wise experiments, not only end-to-end runtime plots,
- states assumptions explicitly and avoids overstating conclusions.

## Canonical context for this folder

- Repository-wide research rules: `agents.md`
- Main theoretical source: `papers/main_gsc/GDE_GSC.pdf`
- SC theory baseline: `papers/von_luxburg_2007_spectral_clustering_tutorial.pdf`
- Main paper manuscript / notation source: `latex/mainv7.tex` (this is the main paper source present in `latex/` in the current checkout)
- Current benchmark script: `complexity_benchmark_clean.py`
- Main manuscript target: `complexity_analysis/complexity.typ`
- Existing outputs in this folder:
  - `complexity_analysis/complexity_pointcloud_comparison.pdf`
  - `complexity_analysis/complexity_network.pdf`
  - `complexity_analysis/complexity_summary.pdf`
  - `complexity_analysis/pointcloud_sc_vs_gsc.csv`
  - `complexity_analysis/network_complexity.csv`

When complexity claims depend on implementation details, inspect the canonical GSC/SC pipeline in:

- `scikit-learn/sklearn/cluster/_spectral.py`
- `scikit-learn/sklearn/manifold/_spectral_embedding.py`
- `scikit-learn/sklearn/manifold/_laplacian.py`
- `competitors/measures.py`
- `competitors/neighbors.py`

## Paper notation and mathematical conventions from `latex/mainv7.tex`

The complexity manuscript must inherit the notation of the main paper, not invent a parallel notation system.

### Object classes and typography

- Graph-level objects and sets use calligraphic notation: `cal(G)`, `cal(V)`, `cal(E)`.
- Matrices and linear operators use bold uppercase Roman symbols: `W`, `P`, `L`, `U`, `Q`, `D`, `I`.
- Scalar entries and scalar-valued functions use lowercase symbols: `w(i, j)`, `p(i, j)`, `f(i)`, `q(i, j)`, `nu(i)`, `pi(i)`.
- Point-cloud data are written as a bold collection `bold(X) = {x_i}_{i=1}^N`, with point vectors `x_i in R^d`.
- Euclidean spaces and probability sets use blackboard-bold notation: `R`, `R_+`, `P`, `E`.
- Standard operators are upright: `diag`, `tr`, `arg min`, `arg max`, `Var`, `Cov`, `SE`, `sign`.
- Transpose is written as a superscript `T` in an upright style; do not switch to prime notation.
- When writing in Typst, preserve these symbol choices and semantic roles even though the exact syntax differs from LaTeX.

### Core data and index conventions

- Graph input is a weighted directed graph `cal(G) = (cal(V), cal(E), w)` by default.
- The default graph assumption in the paper is weak connectivity unless stronger assumptions are explicitly stated.
- `N = |cal(V)|` is the number of vertices; for point clouds it is also the number of samples.
- `d` is the ambient dimension of point-cloud data.
- `i, j, z` denote vertex or sample indices; ordered pair `(i, j)` means the directed edge `i -> j`.
- `k` is the number of clusters and, in the spectral step, the number of requested eigenvectors.
- `kappa` is the cluster index in partitions `V_kappa`.
- `t in N` is the diffusion-time parameter.
- `alpha in R_+` is the measure reweighting exponent.
- `K` is the nearest-neighbor graph-construction parameter in the paper's point-cloud experiments; reserve uppercase `K` for this quantity and lowercase `k` for the number of clusters.

### Point-cloud conventions from the main paper

- A point cloud is written as `bold(X) = {x_i}_{i=1}^N`, with `x_i in R^d`.
- The empirical paper pipeline constructs a sparse directed `K`-nearest-neighbor graph from the point cloud.
- In that setup, `K = ceil(log N)`.
- The paper uses an unweighted asymmetric adjacency matrix `W = {w_ij}` with
  `w_ij = 1{ ||x_i - x_j||^2 / dist_K(x_i)^2 <= 1 }`.
- If the complexity manuscript studies other graph-construction regimes as auxiliary experiments, explicitly mark them as extensions beyond the exact paper protocol.

### Graph, matrix, and measure conventions

- The adjacency matrix is `W = {w_ij}_{i,j=1}^N in R_+^(N x N)`, with `w_ij = w(i, j)`.
- Out-degree and in-degree are `d_out(i) = sum_j w_ij` and `d_in(i) = sum_j w_ji`.
- The transition matrix is `P = D_out^(-1) W`, with entries `p(i, j)`; it is row-stochastic.
- For undirected graphs, `d_out(i) = d_in(i) = d(i)` and the degree matrix may be written `D_d`.
- Symmetrization is written `W_sym = 1/2 (W + W^T)`.
- For a vertex measure `nu`, the associated diagonal matrix is `D_nu = diag(nu)`.
- A positive vertex measure is a map `nu: cal(V) -> R_+`; if `sum_i nu(i) = 1`, it is a probability vertex measure.
- A positive edge measure is `q: cal(V) x cal(V) -> R_+`, with matrix form `Q = [q(i, j)]`.
- The factorized edge-measure convention is `q(i, j) = nu(i) p(i, j)`.
- The incoming measure is `xi = P^T nu`.
- The paper writes `D_(nu+xi) = D_nu + D_xi`.
- When a claim depends on sparsity, write it first in terms of `nnz(W)`; if an implementation or benchmark script uses `A` instead, state once that `A` is the input adjacency and then keep the notation fixed.

### Sets, indicators, and graph functions

- A subset of vertices is denoted by an uppercase Roman letter such as `S` or `U`.
- The complement convention is `bar(S) = cal(V) \ S`; prefer `bar(S)` consistently.
- The paper later contains a `Bar(S)` inconsistency; do not propagate it into the Typst manuscript.
- Characteristic vectors use `chi_S in {0, 1}^N`, with `chi_S(i) = 1` iff `i in S`.
- Singleton basis vectors use `delta_v in {0, 1}^N`.
- The all-ones matrix/vector is written with `1`, for example `1_(N x M)` and `1_(N x 1)`.
- The scalar indicator function is `1{...}`.
- Graph functions are written as `f: cal(V) -> R`, represented by the column vector `f = [f(i)]_(i in cal(V))^T in R^N`.

### Inner products, norms, and function spaces

- Weighted graph-function space is `ell^2(cal(V), nu)`.
- The weighted inner product is `<f, g>_nu = f^T D_nu g = sum_i nu(i) f(i) g(i)`.
- The standard dot product is recovered when `nu = 1_(N x 1)`.
- The paper uses weighted norms induced by `D_(nu+xi)`, for example `||f||_(nu+xi)^2 = <f, D_(nu+xi) f>`.
- When discussing eigensolvers for generalized operators, keep the relevant weighted inner product explicit if it matters for the statement.

### Random-walk conventions

- The random walk is the homogeneous Markov chain `cal(X) = (X_t)_(t >= 0)` on state space `cal(V)`.
- Transition probabilities are `p(i, j) = P(X_(t+1) = j | X_t = i)`.
- Time-marginal distributions are written `p_t(i, .) = delta_i^T P^t`.
- The stationary distribution is `pi in R_+^N` when the walk is strongly connected and aperiodic.
- Reversibility is written `pi(i) p(i, j) = pi(j) p(j, i)`.
- For undirected or reversible walks, `pi(i) propto d(i)`.

### Dirichlet-energy conventions

- Classical Dirichlet energy is written `D^2(f)` or, in parameterized form, `D^2_(nu)(f)` depending on context.
- The paper's macro semantics are:
  - `D{}{f}` for the classical Dirichlet energy associated with `pi`,
  - `D_{nu}^2(f)` for the generalized energy with measure `nu`,
  - `bar(D)_{nu}^2(f)` for the normalized generalized energy.
- The classical energy is
  `D{}{f} = sum_(i,j) pi(i) p(i, j) [f(i) - f(j)]^2`.
- The generalized energy is
  `D_{nu}^2(f) = sum_(i,j) nu(i) p(i, j) [f(i) - f(j)]^2`.
- The normalized generalized energy is
  `bar(D)_{nu}^2(f) = D_{nu}^2(f) / ||f||_(nu+xi)^2`.
- Use the phrase "generalized Dirichlet energy (GDE)" on first use, then "GDE".

### Laplacian and operator conventions

- Classical directed Laplacians are:
  - random-walk Laplacian `L_(RW)`,
  - unnormalized Laplacian `L`,
  - normalized Laplacian `bar(L)`.
- Their paper definitions are:
  - `L_(RW) = I - 1/2 (P + D_pi^(-1) P^T D_pi)`,
  - `L = D_pi - 1/2 (D_pi P + P^T D_pi)`,
  - `bar(L) = D_pi^(-1/2) L D_pi^(-1/2)`.
- Generalized Laplacians are:
  - `L_nu = D_(nu+xi) - (D_nu P + P^T D_nu)`,
  - `L_(RW,nu) = D_(nu+xi)^(-1) L_nu`,
  - `bar(L)_nu = D_(nu+xi)^(-1/2) L_nu D_(nu+xi)^(-1/2)`.
- When `nu = pi`, the paper states `L_pi = 2 L`.
- `L_(RW,nu)` is self-adjoint in `ell^2(cal(V), nu + xi)`.
- `L_nu` and `bar(L)_nu` are self-adjoint in `ell^2(cal(V), nu)`; keep this point explicit when discussing spectral theory or solver assumptions.

### Spectral and optimization conventions

- The Rayleigh quotient is written `R_L(y) = (y^T L y) / (y^T y)` in the classical setting and with the weighted denominator in the generalized setting.
- The main spectral-clustering optimization is written with `tr`, not an ad hoc alternate notation.
- A graph `k`-partition is `bold(V) = {V_kappa}_{kappa=1}^k`.
- Indicator vectors are `u_kappa = chi_(V_kappa)`.
- The indicator matrix is `U = [u_1 ... u_k] in R^(N x k)`.
- The discrete objective is `min tr(U^T L_nu U)` under indicator constraints.
- The relaxed objective is `min tr(U^T L_nu U)` subject to `U^T U = I_k`.
- The clustering step uses the `k` eigenvectors associated with the smallest eigenvalues of the chosen Laplacian.
- The paper states eigenvalues in the Courant-Fischer recall as `lambda_1 >= ... >= lambda_N`; if you discuss smallest-eigenvalue computation for clustering, state the ordering convention explicitly to avoid ambiguity.
- Node embeddings are row-wise: vertex `i` is embedded by the `i`-th row of `U` or `U_(t,alpha)` in `R^k`.

### GSC-specific parameter conventions

- The base diffusion measure is `nu_t = (P^t)^T (1/N) 1_(N x 1)`.
- The parametrized measure is `nu_(t,alpha) = ((P^t)^T (1/N) 1_(N x 1))^(odot alpha)`.
- The associated generalized Laplacian is denoted `L_(t,alpha)` when the dependence on `nu_(t,alpha)` is emphasized.
- In the manuscript, `(t, alpha)` are the canonical GSC hyperparameters.
- If a complexity claim treats `t` or `alpha` as constants, say so explicitly.
- If a complexity experiment varies `t`, note that the cost of forming `nu_t` may scale with repeated sparse matrix-vector multiplications or another implementation-specific strategy.

### Algorithm naming conventions

- `GSC` is the general method family.
- `GSC_un(t, alpha)` uses the unnormalized generalized Laplacian `L_nu`.
- `GSC_n(t, alpha)` uses the normalized generalized Laplacian `bar(L)_nu`.
- `SC_un` and `SC_n` are the classical spectral-clustering baselines on the symmetrized adjacency matrix.
- When comparing SC and GSC, phrase the distinction as "standard Laplacian on symmetrized adjacency" versus "generalized Laplacian induced by the vertex measure `nu` or `nu_(t,alpha)`".

### Complexity-writing rules implied by the paper notation

- Use `N` for sample/node count, `d` for ambient dimension, `K` for nearest-neighbor count, `k` for number of clusters/eigenvectors, `nnz(W)` for sparsity, `t` for diffusion depth, and `alpha` for reweighting.
- Distinguish carefully between the data matrix `X in R^(N x d)`, the adjacency/operator matrices `W, P, L in R^(N x N)`, and the embedding matrix `U in R^(N x k)`.
- For point-cloud input, present the pipeline as: build `W` from `X`, then form the appropriate Laplacian, then compute the spectral embedding, then cluster the rows of `U`.
- For precomputed-network input, treat the adjacency matrix as given; prefer the manuscript notation `W`, and mention `A` only when discussing implementation-specific code.
- When discussing SC overhead, include symmetrization through `W_sym = 1/2 (W + W^T)`.
- When discussing GSC overhead, include the construction of `nu_t`, `nu_(t,alpha)`, `xi = P^T nu`, and the generalized Laplacian.
- If you introduce additional symbols not used in the main paper, define them immediately and justify why they are needed for the complexity discussion.
- Do not switch notation mid-document between `K` and another nearest-neighbor symbol, between `W` and `A`, or between `V` and `C` for clusters without an explicit note.

## Scope of the complexity analysis

The analysis in this folder should be structured around two distinct regimes.

### 1. Point-cloud input

Input is a data matrix `X in R^(N x d)` and the algorithm must first build an affinity graph.

The write-up should separate at least these stages:

1. neighborhood search / graph construction,
2. affinity-matrix assembly,
3. SC symmetrization or GSC measure-related preprocessing,
4. Laplacian construction,
5. eigensolver cost,
6. label extraction (`k`-means or other assignment step).

State clearly which regime is being analyzed:

- dense pairwise-distance construction,
- sparse `k`-NN construction,
- manuscript-style `K`-NN construction with `K = Theta(log N)` or another explicitly stated scaling law,
- fixed ambient dimension `d` versus growing `d`.

### 2. Precomputed-network input

Input is an adjacency matrix `W`, typically sparse.

The write-up should separate at least these stages:

1. optional symmetrization for SC,
2. GSC measure construction / stationary or teleportation-related quantities,
3. generalized Laplacian construction,
4. sparse eigensolver iterations,
5. label extraction.

Make assumptions explicit:

- directed vs undirected graph,
- weighted vs unweighted graph,
- sparse vs dense adjacency,
- connectedness / irreducibility / ergodicity assumptions where needed,
- whether the number of requested eigenvectors/clusters is treated as constant.

## Required scientific posture

- Do not claim that GSC and SC have identical runtime in every sense; claim same asymptotic order only when the assumptions truly match.
- Always distinguish dominant-term asymptotics from lower-order or constant-factor GSC overhead.
- If a GSC-specific object depends on extra parameters (for example `t`, `alpha`, or a power-iteration-like procedure), state whether those are treated as constants or variables in the complexity model.
- If a statement depends on sparsity, write the dependence in terms of `nnz(W)` or graph degree before simplifying to `Theta(N log N)` or another regime.
- If an argument uses ARPACK/Lanczos-style reasoning, say so and state the dependence on matrix-vector products and iteration count carefully.
- If implementation behavior differs from ideal theory, document both: theoretical cost model and observed practical runtime.

## Experimental requirements for this folder

Existing end-to-end runtime plots are not enough on their own. Add component-level evidence whenever possible.

Recommended component-wise measurements:

- graph construction time,
- measure computation time,
- Laplacian assembly time,
- eigensolver time,
- label-assignment time,
- total wall-clock time.

Recommended ablations:

- SC vs GSC on the same point clouds,
- SC vs GSC on the same sparse networks,
- varying `N`,
- varying `d` for point clouds,
- varying `K`,
- varying sparsity / average degree,
- varying GSC-specific parameters only when that affects cost modeling.

For each experiment, record and report:

- seed policy,
- number of repetitions,
- hardware-sensitive caveats if known,
- exact parameter scaling laws,
- whether plotted runtime is mean, median, or another statistic,
- what quantity is on each axis and why it tests the claimed complexity law.

## Typst: non-negotiable rules

These rules come from the user request and must be followed for work in this folder.

1. Always output Typst, never raw LaTeX.
2. If unsure about syntax, always check Typst documentation first.
3. Do not invent unreadable or uncertain parts; mark uncertain fragments clearly.
4. Match existing repository style before introducing alternatives.
5. When using images as sources, extract the mathematical/document content only and ignore visual decoration such as page layout, colors, and background styling.

## Official Typst docs (check these first)

- Main docs: https://typst.app/docs/
- Syntax reference (markup / math / code): https://typst.app/docs/reference/syntax/
- Math reference: https://typst.app/docs/reference/math/
- Equation reference: https://typst.app/docs/reference/math/equation/
- Numbered lists (`+ item`): https://typst.app/docs/reference/model/enum/
- Bullet lists (`- item`): https://typst.app/docs/reference/model/list/
- Symbols reference: https://typst.app/docs/reference/symbols/

## Typst writing conventions for this manuscript

- `complexity_analysis/complexity.typ` is the authoritative manuscript file for this task.
- Write native Typst, not LaTeX disguised as Typst.
- Keep notation aligned with the main GSC paper and the repository-wide conventions:
  - prefer the paper's adjacency notation `W` over a generic `A`, with transition matrix `P = D_out^(-1) W`, generalized measure `nu`, auxiliary term `xi`, and the Laplacian symbols used by the paper,
  - clear distinction between SC (`standard=True`) and GSC (`standard=False` with active measure).
- Prefer the paper's actual symbol choices when possible: `W` for adjacency, `K` for nearest-neighbor count, `k` for clusters/eigenvectors, `bold(V)` for partitions, and `U` for the spectral embedding matrix.
- Every displayed formula should be introduced in prose and all symbols should be defined nearby.
- Every theorem-like claim should list its assumptions before the conclusion.
- Every complexity statement should specify the relevant variables (`N`, `d`, `K`, `k`, `nnz`, iteration counts, diffusion depth `t`, and any solver-iteration parameters that matter).
- Prefer explicit caveats over overconfident prose. If a syntax fragment or theoretical step is uncertain, mark it clearly and resolve it before finalizing.

## Suggested manuscript structure

The eventual Typst document should normally include:

1. problem setup and notation,
2. algorithmic decomposition of SC and GSC,
3. theoretical complexity analysis by stage,
4. point-cloud complexity regime,
5. network complexity regime,
6. component-wise empirical validation,
7. end-to-end empirical validation,
8. discussion of dominant terms, overheads, and limitations,
9. concise conclusion.

## File/output conventions

- Keep manuscript sources in `complexity_analysis/`.
- Keep figure names descriptive and stable.
- If new benchmark scripts are created for this task, ensure their outputs can be copied or saved into `complexity_analysis/` for manuscript use.
- Do not silently replace existing figures; prefer additive filenames unless the replacement is intentional and documented.

## Completion checklist for work in this folder

- [ ] Root `agents.md` was read first.
- [ ] Main GSC paper and SC reference were consulted for the claims being made.
- [ ] All asymptotic claims state their assumptions.
- [ ] The manuscript distinguishes theory from measured runtime.
- [ ] Component-level experiments complement the end-to-end plots.
- [ ] Typst syntax follows official docs rather than LaTeX habits.
- [ ] Notation matches the repository and the main GSC paper.
