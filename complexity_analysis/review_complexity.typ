= Complexity Comparison: SC and GSC

#import "@preview/algorithmic:1.0.7"
#import algorithmic: algorithm-figure, style-algorithm

#show: style-algorithm
#show bibliography: set heading(numbering: none)

We consider a sparse adjacency matrix $W$ with $N$ vertices and $m$ stored edges. Under the classical sparse-graph convention, one has $N = O(m)$, so $O(m + N)$ and $O(m)$ are equivalent. #linebreak()  
We compare the time complexity of a single run of the GSC algorithm and the classical SC algorithm on $W$. The implementation and experiments considered below use the normalized Laplacian path; however, the preprocessing complexity analysis given here remains valid for both normalized and unnormalized Laplacians.
== GSC overhead
The additional GSC-specific operations are the computation of the measure $nu_(t,alpha)$ and the assembly of the generalized Laplacian $L_(nu_(t,alpha))$ or its normalized counterpart $macron(L)_(nu_(t,alpha))$.
The GSC measure used in the implementation is computed by repeated sparse multiplication with the transition matrix $P = D_"out"^(-1) W$, without ever forming the dense power $P^t$ explicitly.

#algorithm-figure([$(alpha,t)$-dependent measure computation], {
  import algorithmic: *
  Procedure(
    "Compute-Measure",
    ($W$, $t$, $alpha$),
    {
      Assign[$P$][$D_"out"^(-1) W$]
      Assign[$v_0$][$1 / N 1_(1 times N)$]
      For(
        [$ell = 0, 1, dots, t - 1$],
        {
          Assign[$v_(ell+1)$][$v_ell P$]
        },
      )
      Assign[$nu_t$][$v_t^T$]
      Assign[$nu_(t, alpha)(i)$][$nu_t(i)^alpha$]
      Return[$nu_(t, alpha)$]
    },
  )
}) <alg-measure>

Since $P$ has the same sparsity as $W$, constructing it costs $O(m + N)$. Each update $v_(ell+1) = v_ell P$ is a sparse vector-matrix multiplication with cost $O(m)$, hence the $t$ iterations cost $O(t m)$. The final componentwise power and normalization cost $O(N)$. Therefore the total complexity of the measure computation is $O(t m + m + N)$, which is $O(m)$ for fixed $t$ under the classical sparse-graph convention.


== Generalized Laplacians building cost and sparsity
We now prove that the generalized Laplacian has the same sparsity order as the classical undirected Laplacian, and the asymptotic cost of building them is the same. Let

$
W_"sym" = 1/2 (W + W^T) quad "and" quad L_"sym" = D_"sym" - W_"sym"
$

On the GSC side, recall that the generalized Laplacian is

$
L_nu = D_(nu + xi) - (D_nu P + P^T D_nu),
$

It is easy to see that both Laplacians have the same sparsity order as $W_"sym"$.
Indeed, left and right multiplication by diagonal matrices only rescales existing entries and cannot create new off-diagonal nonzeros. Hence $L_"sym"$ and $L_(nu_(t,alpha))$ store only diagonal terms together with entries corresponding to the support of $W$ and $W^T$, so their storage and assembly costs remain linear in the number of graph edges, i.e. $O(m)$. The same statement holds for their normalized counterparts.

In the normalized implementation path considered here, both the standard Laplacian built from $W_"sym"$ and the normalized generalized Laplacian are symmetric. Therefore the same symmetric ARPACK routine is used for the spectral decomposition of the benchmarked SC and GSC variants, with the same asymptotic model for the spectral step.


Finally, we conclude that, for fixed $t$, the preprocessing overhead of a single run of GSC over SC is linear in the number of edges in the graph.
== Spectral Decomposition
In practice, the spectral step is carried out by `eigsh`, which wraps ARPACK's implicitly restarted Lanczos method. In the normalized scikit-learn path benchmarked here, it is used in shift-invert mode, so the implementation-level cost depends on sparse factorizations, repeated linear solves, restart parameters, and convergence behavior rather than on a single fixed formula @scipy-arpack-tutorial @scipy-eigsh @arpack1998.

For this reason, we do not claim a clean implementation-level complexity law for the spectral step. The important point for the present comparison is that the benchmarked SC and GSC variants use the same normalized spectral routine, and therefore share the same dominant spectral cost @scipy-arpack-tutorial @scipy-eigsh @arpack1998.
== K-Means
Finally, the label extraction step is performed by `kmeans` on the $N times r$ spectral embedding. The scikit-learn documentation gives the average complexity of $k$-means as $O(k n T)$, where $k$ is the number of clusters, $n$ the number of samples, and $T$ the number of Lloyd iterations @sklearn-kmeans. In the present setting, both the embedding dimension and the number of clusters are $r$, and with $s$ restarts and $I$ Lloyd iterations this yields the model $O(s I N r^2)$. This term is the same for SC and GSC, and is $O(N)$ for fixed $r$, $s$, and $I$.
== Full Single-Run Complexity
At the level of a single run, SC and GSC have the same complexity up to the additional GSC preprocessing term analyzed above. Thus the only systematic difference between the two methods is the cost of building the measure and the generalized Laplacian, while the spectral and labeling steps are shared.

= Experimental results

The experiments are carried out on sparse DISBM networks with expected degree of order $Theta(log N)$, hence $m = Theta(N log N)$ asymptotically. We first validate the part of the theory that is proved explicitly, namely the preprocessing overhead.  @fig-network-pre shows that the preprocessing cost of both SC and GSC is consistent with linear sparse scaling in $m$, with a larger constant for GSC. We then compare the full algorithms in  @fig-network-sc-gsc. SC and GSC exhibit the same empirical growth, and their end-to-end runtime difference remains small compared with the total cost.  @fig-network-components explains this behavior: the eigensolver rapidly dominates the runtime, while the preprocessing terms become negligible. Thus the experiments support both claims needed here: the preprocessing analysis is validated directly, and the full algorithms have the same effective scaling because they share the same dominant spectral step.

#figure(
  image("experiments/figures/network_preprocessing_sc_vs_gsc.png", width: 100%),
  caption: [Preprocessing-only benchmark on sparse precomputed DISBM networks. The SC and GSC preprocessing costs scale linearly with the sparse graph size up to constants, with GSC carrying the larger constant factor predicted by the theory.],
) <fig-network-pre>

#figure(
  image("experiments/figures/network_sc_vs_gsc_single_plot.png", width: 100%),
  caption: [Single-run SC and GSC runtimes on sparse precomputed DISBM networks with expected degree of order $Theta(log N)$. The left panel shows that the two methods have the same empirical growth, while the right panel reports the end-to-end runtime difference $"GSC" - "SC"$.],
) <fig-network-sc-gsc>

#figure(
  image("experiments/figures/network_component_share.png", width: 100%),
  caption: [Component dominance on sparse precomputed DISBM networks. The eigensolver rapidly dominates the runtime, while both SC and GSC preprocessing terms become negligible in relative magnitude.],
) <fig-network-components>

#bibliography("references.bib", style: "ieee")
