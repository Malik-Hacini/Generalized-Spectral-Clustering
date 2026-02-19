# C-L Reciprocity for Binary KNN Graphs: Optimized Derivation

## Context

In GSC, point-cloud datasets are embedded as **directed** K-nearest-neighbor (KNN) graphs.
The adjacency matrix $\mathbf{W}$ is binary: $W_{ij} = 1$ iff $j$ is among $i$'s $K$ nearest neighbors.
Standard SC symmetrizes via $\frac{1}{2}(\mathbf{W} + \mathbf{W}^\top)$; GSC works directly with the directed $\mathbf{W}$.

This document derives efficient O(NK) algorithms for reciprocity and in-degree analysis.

---

## 1. KNN Graph Properties

Given $N$ points and neighbor count $K = \lceil \log N \rceil$:

- **Out-degree**: Every node has exactly $K$ outgoing edges $\Rightarrow d_i^{\text{out}} = K \; \forall i$.
- **In-degree**: $d_j^{\text{in}} = \sum_i W_{ij}$ varies across nodes. A node that is a "hub" (close to many others) has high in-degree.
- **Total edges**: $\text{nnz}(\mathbf{W}) = NK$.
- **Sparsity**: $\mathbf{W}$ is a `scipy.sparse.csr_matrix` with $NK$ stored entries.

The **in-degree distribution** characterizes the graph's asymmetry. If all edges were reciprocated, in-degree would equal out-degree $= K$ for all nodes. Deviation from this reveals structural asymmetry.

---

## 2. Standard Reciprocity (Eq. 12)

$$\text{Reciprocity}(\mathbf{W}) = \frac{\sum_{i,j} W_{ij} \cdot W_{ji}}{\sum_{i,j} W_{ij}}$$

### Simplification for binary KNN

Since $W_{ij} \in \{0,1\}$, the numerator counts **reciprocated edges** (pairs where both $i \to j$ and $j \to i$ exist):

$$\text{Reciprocity}(\mathbf{W}) = \frac{|\{(i,j) : W_{ij}=1 \wedge W_{ji}=1\}|}{NK}$$

### Efficient computation

Using element-wise multiplication of sparse matrices:

$$\text{numerator} = \mathbf{W} \circ \mathbf{W}^\top \quad \text{(Hadamard product)}$$

```python
reciprocated = W.multiply(W.T).sum()
reciprocity = reciprocated / W.sum()
```

**Complexity**: $O(\text{nnz}) = O(NK)$ for the sparse Hadamard product.

---

## 3. C-L Reciprocity (Eq. 14)

$$\text{C-L Reciprocity}(\mathbf{W}, y) = \frac{1}{k(k-1)} \sum_{a \neq b} \frac{2 \cdot \min(E(a,b), E(b,a))}{E(a,b) + E(b,a)}$$

where $E(a,b) = \sum_{i \in C_a, j \in C_b} W_{ij}$ is the number of directed edges from cluster $a$ to cluster $b$, $y$ is the cluster assignment, and $k$ is the number of clusters.

### Step 1: Inter-cluster edge matrix $\mathbf{E}$ (size $k \times k$)

Instead of summing over all $N^2$ pairs, iterate only over the $NK$ nonzero entries of $\mathbf{W}$:

$$E[a, b] = \sum_{(i,j) \in \text{nnz}(\mathbf{W})} \mathbf{1}[y_i = a] \cdot \mathbf{1}[y_j = b]$$

**Efficient matrix formulation**: Let $\mathbf{Y} \in \{0,1\}^{N \times k}$ be the one-hot cluster indicator matrix. Then:

$$\mathbf{E} = \mathbf{Y}^\top \mathbf{W} \mathbf{Y}$$

This is a sparse-dense product chain: $O(NK \cdot k)$ total, which for $k \ll N$ is effectively $O(NK)$.

### Step 2: Pairwise reciprocity from $\mathbf{E}$

For each unordered pair $\{a, b\}$ with $a \neq b$:

$$r_{ab} = \frac{2 \cdot \min(E_{ab}, E_{ba})}{E_{ab} + E_{ba}}$$

with the convention $r_{ab} = 0$ when $E_{ab} + E_{ba} = 0$ (no edges between clusters $a$ and $b$).

Note: $r_{ab} = r_{ba}$, so summing over ordered pairs $a \neq b$ and dividing by $k(k-1)$ is equivalent to summing over unordered pairs and dividing by $\binom{k}{2}$... but we follow the paper's convention of ordered pairs.

### Step 3: Average

$$\text{C-L Reciprocity} = \frac{1}{k(k-1)} \sum_{a \neq b} r_{ab}$$

Since $r_{ab} = r_{ba}$, each unordered pair contributes twice, so:

$$\text{C-L Reciprocity} = \frac{1}{\binom{k}{2}} \sum_{a < b} r_{ab}$$

**Complexity**: $O(k^2)$ for the summation — negligible since $k \leq 15$ for UCI datasets.

---

## 4. Total Algorithm Complexity

| Step | Operation | Complexity |
|------|-----------|------------|
| Build KNN graph | `kneighbors_graph(X, K)` | $O(DNK)$ via ball tree |
| In-degree distribution | `W.T.sum(axis=1)` or `W.sum(axis=0)` | $O(NK)$ |
| Standard reciprocity | `W.multiply(W.T).sum() / W.sum()` | $O(NK)$ |
| Inter-cluster edge matrix | $\mathbf{Y}^\top \mathbf{W} \mathbf{Y}$ | $O(NK)$ |
| C-L reciprocity from $\mathbf{E}$ | Double loop over $k$ clusters | $O(k^2)$ |
| **Total** | | **$O(DNK)$** |

For the UCI datasets ($N \leq 5000$, $D \leq 60$, $K \leq 9$), each dataset completes in milliseconds.

---

## 5. Interpretation

- **Standard Reciprocity** $\in [0, 1]$: Fraction of directed edges that are reciprocated. Higher = more symmetric graph. In KNN, larger $K$ relative to $N$ tends to increase reciprocity.

- **C-L Reciprocity** $\in [0, 1]$: Measures whether inter-cluster edges are balanced directionally. A value of 1 means $E(a,b) = E(b,a)$ for all cluster pairs — the directed graph is "cluster-symmetric". Low values indicate directional bias between clusters.

- **In-degree distribution**: If concentrated around $K$, the graph is nearly symmetric. Heavy tails (high in-degree hubs or isolated nodes with in-degree 0) indicate strong asymmetry — precisely the structure that GSC's directed Laplacian can exploit.
