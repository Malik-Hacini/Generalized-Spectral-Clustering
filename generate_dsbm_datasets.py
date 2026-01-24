"""
Asymmetry-Controlled DSBM Dataset Generation
=============================================

This script generates synthetic directed graphs with controlled asymmetry levels
for benchmarking spectral clustering algorithms on directed graphs.

Mathematical Framework:
- Meta-graph: F(η) = F_base + η · F_flow
- F_base: Symmetric (strong intra-cluster, weak inter-cluster)
- F_flow: Antisymmetric cyclic flow pattern
- η ∈ [0, 1]: Asymmetry parameter

Parameters chosen for detectable clusters:
- p_in = 0.8 (high intra-cluster density)
- p_out = 0.1 (low inter-cluster density)  
- Ratio = 8x (clearly separable clusters)
"""

import os
import json
import numpy as np
import scipy.sparse as sp
from torch_geometric_signed_directed.data import DSBM


def save_dsbm_dataset(adj_matrix: sp.csr_matrix, labels: np.ndarray, 
                      save_path: str, dataset_name: str):
    """Save a DSBM dataset in the format expected by load_dataset()."""
    dataset_dir = os.path.join(save_path, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    if not isinstance(adj_matrix, sp.csr_matrix):
        adj_matrix = sp.csr_matrix(adj_matrix)
    
    np.savez(
        os.path.join(dataset_dir, 'graph.npz'),
        adj_data=adj_matrix.data,
        adj_indices=adj_matrix.indices,
        adj_indptr=adj_matrix.indptr,
        adj_shape=np.array(adj_matrix.shape),
        labels=labels
    )


def compute_asymmetry(adj: sp.csr_matrix) -> float:
    """Compute asymmetry: ||A - A^T||_F / ||A + A^T||_F"""
    A = adj.toarray().astype(float)
    norm_sym = np.linalg.norm(A + A.T, 'fro')
    norm_antisym = np.linalg.norm(A - A.T, 'fro')
    if norm_sym < 1e-10:
        return 1.0 if norm_antisym > 1e-10 else 0.0
    return norm_antisym / norm_sym


def build_meta_graph(K: int, p_in: float, p_out: float, delta: float, eta: float) -> np.ndarray:
    """
    Build meta-graph matrix F with controlled asymmetry.
    F(η) = F_base + η · F_flow
    """
    # F_base: symmetric with strong diagonal (intra-cluster)
    F_base = np.full((K, K), p_out)
    np.fill_diagonal(F_base, p_in)
    
    # F_flow: antisymmetric cyclic flow (0 -> 1 -> 2 -> ... -> 0)
    F_flow = np.zeros((K, K))
    for i in range(K):
        j = (i + 1) % K
        F_flow[i, j] = delta
        F_flow[j, i] = -delta
    
    F = F_base + eta * F_flow
    return np.clip(F, 0.0, 1.0)


def generate_asymmetry_controlled_datasets(
    N: int = 500,
    K: int = 3,
    p: float = 0.05,
    p_in: float = 0.8,
    p_out: float = 0.1,
    delta: float = 0.3,
    eta_values: list = None,
    seeds: list = None,
    base_dir: str = "DSBM_datasets"
):
    """Generate DSBM datasets with varying asymmetry levels."""
    
    if eta_values is None:
        eta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    if seeds is None:
        seeds = [42, 123, 456]
    
    os.makedirs(base_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING ASYMMETRY-CONTROLLED DSBM DATASETS")
    print("=" * 70)
    print(f"\nFixed parameters:")
    print(f"  N = {N} (nodes), K = {K} (clusters), p = {p} (density)")
    print(f"  p_in = {p_in}, p_out = {p_out} (ratio = {p_in/p_out:.1f}x)")
    print(f"  δ = {delta} (flow strength)")
    print(f"\nη values: {eta_values}")
    print(f"Seeds: {seeds}\n")
    
    results = []
    
    for eta in eta_values:
        print("-" * 70)
        print(f"η = {eta:.1f}")
        
        F = build_meta_graph(K, p_in, p_out, delta, eta)
        print(f"F matrix:\n{np.round(F, 2)}")
        
        for seed in seeds:
            np.random.seed(seed)
            A, labels = DSBM(N, K, p, F, size_ratio=1.0)
            asymmetry = compute_asymmetry(A)
            name = f"dsbm_eta{eta:.1f}_seed{seed}"
            
            print(f"  {name}: edges={A.nnz}, asymmetry={asymmetry:.4f}")
            save_dsbm_dataset(A, labels, base_dir, name)
            
            results.append({
                'eta': eta, 'seed': seed, 'name': name,
                'edges': int(A.nnz), 'asymmetry': float(asymmetry)
            })
    
    # Save metadata
    metadata = {
        'params': {'N': N, 'K': K, 'p': p, 'p_in': p_in, 'p_out': p_out, 'delta': delta},
        'eta_values': eta_values, 'seeds': seeds, 'datasets': results
    }
    with open(os.path.join(base_dir, 'asymmetry_experiment_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Generated {len(results)} datasets in {base_dir}/")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    generate_asymmetry_controlled_datasets()
