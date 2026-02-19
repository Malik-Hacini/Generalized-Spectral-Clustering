"""
Asymmetry-Controlled DSBM Dataset Generation (Interpolation Method)
===================================================================

This script generates synthetic directed graphs with controlled asymmetry levels
for benchmarking spectral clustering algorithms on directed graphs.

Mathematical Framework:
- Meta-graph: F(γ) = (1-γ) * F_sym + γ * F_cyclic
- F_sym: Symmetric assortative matrix (easy for all methods)
- F_cyclic: Pure cyclic flow matrix (hard for classical SC, solvable for GSC)
- γ ∈ [0, 1]: Interpolation parameter (0 = pure symmetric, 1 = pure cyclic)

See dsbm_derivation.md for the full mathematical justification.
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
    """Compute empirical asymmetry: ||A - A^T||_F / ||A + A^T||_F"""
    A = adj.toarray().astype(float)
    norm_sym = np.linalg.norm(A + A.T, 'fro')
    norm_antisym = np.linalg.norm(A - A.T, 'fro')
    
    if norm_sym < 1e-10:
        return 1.0 if norm_antisym > 1e-10 else 0.0
        
    return float(norm_antisym / norm_sym)


def build_meta_graph(gamma: float) -> np.ndarray:
    """
    Build meta-graph matrix F(γ) interpolating symmetric and cyclic structure.
    γ ∈ [0, 1]
    
    F(γ) = (1-γ) * F_sym + γ * F_cyclic
    """
    # F_sym: Symmetric with strong diagonal (assortative)
    # Average entry = (0.9*3 + 0.3*6) / 9 = 4.5 / 9 = 0.5
    F_sym = np.array([
        [0.9, 0.3, 0.3],
        [0.3, 0.9, 0.3],
        [0.3, 0.3, 0.9]
    ])
    
    # F_cyclic: Pure flow matrix with uniform symmetric component
    # F_cyclic + F_cyclic^T = J (matrix of ones) -> no symmetric signal
    # Average entry = (0.5*3 + 1.0*3 + 0.0*3) / 9 = 4.5 / 9 = 0.5
    F_cyclic = np.array([
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 1.0],
        [1.0, 0.0, 0.5]
    ])
    
    F = (1.0 - gamma) * F_sym + gamma * F_cyclic
    return np.clip(F, 0.0, 1.0)


def generate_asymmetry_controlled_datasets(
    N: int = 600,
    K: int = 3,
    p: float = 0.05,
    gamma_values: list | None = None,
    seeds: list | None = None,
    base_dir: str = "DSBM_datasets"
):
    """Generate DSBM datasets with varying asymmetry interpolation levels."""
    
    if gamma_values is None:
        gamma_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    if seeds is None:
        seeds = [42, 123, 456, 789, 101112]
    
    os.makedirs(base_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING ASYMMETRY-CONTROLLED DSBM DATASETS")
    print("=" * 70)
    print(f"\nFixed parameters:")
    print(f"  N = {N} (nodes), K = {K} (clusters), p = {p} (base density)")
    print(f"\nγ values: {gamma_values}")
    print(f"Seeds: {seeds}\n")
    
    results = []
    
    for gamma in gamma_values:
        print("-" * 70)
        print(f"γ = {gamma:.1f}")
        
        F = build_meta_graph(gamma)
        print(f"Meta-graph matrix F(γ):\n{np.round(F, 2)}")
        
        for seed in seeds:
            np.random.seed(seed)
            A, labels = DSBM(N, K, p, F, size_ratio=1.0)
            
            # Ensure A is a csr_matrix
            if not isinstance(A, sp.csr_matrix):
                A = sp.csr_matrix(A)
                
            asymmetry = compute_asymmetry(A)
            name = f"dsbm_gamma{gamma:.1f}_seed{seed}"
            
            print(f"  {name}: edges={A.nnz}, empirical asymmetry={asymmetry:.4f}")
            save_dsbm_dataset(A, labels, base_dir, name)
            
            results.append({
                'gamma': float(gamma), 'seed': int(seed), 'name': name,
                'edges': int(A.nnz), 'asymmetry': float(asymmetry)
            })
    
    # Save metadata
    metadata = {
        'params': {'N': N, 'K': K, 'p': p},
        'gamma_values': gamma_values, 'seeds': seeds, 'datasets': results
    }
    with open(os.path.join(base_dir, 'asymmetry_experiment_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Generated {len(results)} datasets in {base_dir}/")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    generate_asymmetry_controlled_datasets()
