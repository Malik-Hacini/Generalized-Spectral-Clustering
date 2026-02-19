# Directed Stochastic Block Model (DSBM) Generation for Asymmetry Benchmarking

**Author:** opencode
**Date:** 2026-02-19
**Context:** Benchmark dataset generation for Generalized Spectral Clustering (GSC).

## 1. The Limitation of Standard Symmetric DSBMs

Classical spectral clustering operates on the symmetrized adjacency matrix, $S = \frac{1}{2}(A + A^T)$. In standard directed stochastic block models (DSBMs), the probability of an edge existing from block $i$ to block $j$ is given by a matrix $F_{ij}$. 

A naive approach to adding directionality is to define $F(\eta) = F_{\text{base}} + \eta F_{\text{flow}}$, where $F_{\text{base}}$ is a symmetric assortative matrix (strong diagonal, weak off-diagonal) and $F_{\text{flow}}$ is a skew-symmetric cyclic matrix (e.g., $F_{\text{flow}} = -F_{\text{flow}}^T$). 

If we examine the expected symmetrized matrix $E[S]$, we find:
$$ E[S] \propto F(\eta) + F(\eta)^T = 2 F_{\text{base}} + \eta (F_{\text{flow}} + F_{\text{flow}}^T) $$
Because $F_{\text{flow}}$ is skew-symmetric, $F_{\text{flow}} + F_{\text{flow}}^T = 0$. Thus, $E[S] \propto 2 F_{\text{base}}$ **for all values of $\eta$**.

**Conclusion:** The symmetric part of the graph retains a massive assortative community signal regardless of how much directionality ($\eta$) is added. Classical spectral clustering will easily identify the clusters using $S$, failing to demonstrate the necessity of directed methods like GSC.

## 2. Theoretical Solution: The Flow-Driven DSBM

To create a dataset that genuinely challenges classical spectral clustering, we must design the meta-graph $F$ such that its symmetric part $F + F^T$ contains **no community structure** (i.e., it is a matrix of constants). The community structure must be encoded entirely in the directed flow.

This specific problem setup was studied in the context of the **DIGRAC algorithm** [1]. They propose the **Cyclic Meta-Graph** for $K$ clusters. For $K=3$, the ideal challenging cyclic probability matrix $F_{\text{cyclic}}$ is:

$$ F_{\text{cyclic}} = \begin{bmatrix} 0.5 & 1.0 & 0.0 \\ 0.0 & 0.5 & 1.0 \\ 1.0 & 0.0 & 0.5 \end{bmatrix} $$

Notice that $F_{\text{cyclic}} + F_{\text{cyclic}}^T = \mathbf{1}\mathbf{1}^T$ (a matrix of all ones). If a dataset is generated with this $F$, the expected undirected edge density between any two nodes is identical. Classical SC will see a pure Erdős-Rényi random graph and will fail completely (Adjusted Rand Index $\approx 0$), while directed methods can leverage the cyclic flows ($1.0$ vs $0.0$) to cluster the nodes.

## 3. The Directionality Interpolation Scheme

To empirically demonstrate that "the more directional the graph, the more classical SC fails", we interpolate smoothly between an "easy" symmetric graph and a "hard" directed graph using a single parameter $\gamma \in [0, 1]$.

We fix the average entry of the matrices to be constant to ensure the global graph sparsity remains roughly constant as $\gamma$ changes.

### Step 1: Base Matrices

**$F_{\text{sym}}$ (Easy for all methods):**
An assortative symmetric matrix with strong community structure.
$$ F_{\text{sym}} = \begin{bmatrix} 0.9 & 0.3 & 0.3 \\ 0.3 & 0.9 & 0.3 \\ 0.3 & 0.3 & 0.9 \end{bmatrix} $$

**$F_{\text{cyclic}}$ (Hard for classical SC, Easy for GSC):**
A flow-based matrix where the symmetric signal is entirely uniform.
$$ F_{\text{cyclic}} = \begin{bmatrix} 0.5 & 1.0 & 0.0 \\ 0.0 & 0.5 & 1.0 \\ 1.0 & 0.0 & 0.5 \end{bmatrix} $$

### Step 2: Interpolation Function

We generate datasets using the meta-graph $F(\gamma)$:
$$ F(\gamma) = (1-\gamma) F_{\text{sym}} + \gamma F_{\text{cyclic}} $$

### Step 3: Theoretical Asymmetry and Signal

The symmetric signal available to classical SC is proportional to the difference between the diagonal (intra-cluster) and the symmetric off-diagonal (inter-cluster) probabilities.
*   **At $\gamma = 0$:** $F(0) = F_{\text{sym}}$. Classical SC has a $0.9 / 0.3 = 3:1$ signal ratio. Easy.
*   **At $\gamma = 0.5$:** 
    $$ F(0.5) = \begin{bmatrix} 0.7 & 0.65 & 0.15 \\ 0.15 & 0.7 & 0.65 \\ 0.65 & 0.15 & 0.7 \end{bmatrix} $$
    The symmetric signal ratio is $0.7 / (\frac{0.65+0.15}{2}) = 0.7 / 0.4 = 1.75:1$. Classical SC performance degrades.
*   **At $\gamma = 1.0$:** $F(1) = F_{\text{cyclic}}$. The symmetric signal ratio is $0.5 / (\frac{1.0+0.0}{2}) = 0.5 / 0.5 = 1:1$ (pure noise). Classical SC fails completely.

Meanwhile, the generalized Dirichlet energy $\nu_{t, \alpha}$ utilized by GSC naturally transitions from separating symmetric assortative blocks ($\gamma=0$) to separating directed cyclic flows ($\gamma=1$), remaining theoretically detectable throughout the interpolation curve.

## 4. Empirical Asymmetry Metric

To validate the generated graphs, we measure the empirical asymmetry of the resulting adjacency matrix $A$:

$$ \text{Asymmetry}(A) = \frac{\| A - A^T \|_F}{\| A + A^T \|_F} $$

This bounded metric $\in [0, 1]$ tracks the interpolation parameter $\gamma$ at the realized graph level.

## References

[1] Y. He, M. Reinert, and M. Cucuringu, "DIGRAC: Digraph Clustering Based on Flow Imbalance," in *Proceedings of the Learning on Graphs Conference (LoG 2022)*, PMLR 198, 2022. Available: https://arxiv.org/abs/2106.05194.
