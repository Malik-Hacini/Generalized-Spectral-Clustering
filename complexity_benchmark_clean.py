#!/usr/bin/env python3
"""
Comprehensive Computational Complexity Analysis for GSC Framework

This script validates:
1. GSC and SC have the same O(N² log N) complexity on point clouds
2. Networks (given adjacency) have O(N² log N) complexity dominated by eigendecomposition

Theoretical Background:
=======================
Von Luxburg (2007) "A Tutorial on Spectral Clustering" establishes:
- Standard SC: O(N³) for dense matrices, O(N × nnz × k) for sparse eigensolvers
- For sparse k-NN graphs with nnz = O(N log N): O(N² log N)

References:
-----------
[1] Von Luxburg, U. (2007). "A Tutorial on Spectral Clustering." 
    Statistics and Computing, 17(4):395-416.
[2] Ng, A.Y., Jordan, M.I., Weiss, Y. (2001). "On Spectral Clustering: 
    Analysis and an Algorithm." NeurIPS 14.
[3] Lehoucq, R.B., Sorensen, D.C., Yang, C. (1998). "ARPACK Users' Guide."
    SIAM Publications.
"""

import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import linregress
from sklearn import cluster
from sklearn.datasets import make_blobs
from competitors.measures import teleporting_undirected_measure
from competitors.neighbors import log_neighbors

# =============================================================================
# Configuration
# =============================================================================

# Point cloud experiments
N_VALUES_POINTCLOUD = [500, 1000, 2000, 5000, 10000, 20000]
D_VALUE = 10  # Fixed dimension for GSC vs SC comparison

# Network experiments  
N_VALUES_NETWORK = [500, 1000, 2000, 5000, 10000]

# Common parameters
K_CLUSTERS = 3
N_ITERATIONS = 3  # Repeat each measurement
RANDOM_STATE = 42

# Output paths
RESULTS_PATH = "results/complexity_analysis"
os.makedirs(RESULTS_PATH, exist_ok=True)

# Plot style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# =============================================================================
# Data Generation
# =============================================================================

def generate_point_cloud(N, D, K=3, seed=42):
    """Generate a Gaussian mixture point cloud."""
    X, y = make_blobs(n_samples=N, n_features=D, centers=K, 
                      cluster_std=2.5, random_state=seed)
    return X.astype(np.float64), y

def generate_sparse_network(N, K=3, avg_degree_factor=2, seed=42):
    """
    Generate a sparse directed network with community structure.
    Returns sparse adjacency matrix and labels.
    
    The network has:
    - N nodes
    - K communities
    - Average degree ~ avg_degree_factor * log(N)
    """
    np.random.seed(seed)
    
    # Assign nodes to communities
    labels = np.repeat(np.arange(K), N // K)
    if len(labels) < N:
        labels = np.concatenate([labels, np.zeros(N - len(labels), dtype=int)])
    np.random.shuffle(labels)
    
    # Generate edges - higher probability within communities
    k = int(avg_degree_factor * np.log(N))  # Target average degree
    p_in = 0.8  # Probability of intra-community edge
    p_out = 0.2  # Probability of inter-community edge
    
    rows, cols = [], []
    for i in range(N):
        # Sample k neighbors with community bias
        probs = np.where(labels == labels[i], p_in, p_out)
        probs[i] = 0  # No self-loops
        probs = probs / probs.sum()
        neighbors = np.random.choice(N, size=min(k, N-1), replace=False, p=probs)
        rows.extend([i] * len(neighbors))
        cols.extend(neighbors)
    
    # Create sparse adjacency matrix
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    
    return adjacency, labels

# =============================================================================
# Benchmarking Functions
# =============================================================================

def benchmark_sc_pointcloud(X, n_clusters, n_neighbors):
    """
    Benchmark Standard Spectral Clustering on point cloud.
    Uses standard=True to symmetrize the affinity matrix.
    """
    start = time.perf_counter()
    sc = cluster.SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        laplacian_method='norm',  # Normalized Laplacian
        standard=True,            # Standard SC (symmetrizes)
        measure=None,             # No custom measure
        assign_labels='kmeans',
        random_state=RANDOM_STATE
    )
    sc.fit(X)
    elapsed = time.perf_counter() - start
    return elapsed

def benchmark_gsc_pointcloud(X, n_clusters, n_neighbors, t=5, alpha=0.5):
    """
    Benchmark Generalized Spectral Clustering on point cloud.
    Uses standard=False (no symmetrization) with custom measure.
    The measure is passed as (callable, kwargs) tuple - the pipeline resolves it.
    """
    start = time.perf_counter()
    sc = cluster.SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        laplacian_method='norm',  # Normalized Laplacian
        standard=False,           # GSC (no symmetrization)
        measure=(teleporting_undirected_measure, {'alpha': alpha, 't': t}),
        assign_labels='kmeans',
        random_state=RANDOM_STATE
    )
    sc.fit(X)
    elapsed = time.perf_counter() - start
    return elapsed

def benchmark_sc_network(adjacency, n_clusters):
    """
    Benchmark Standard Spectral Clustering on network (precomputed adjacency).
    """
    start = time.perf_counter()
    sc = cluster.SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        laplacian_method='norm',
        standard=True,            # Symmetrize
        measure=None,
        assign_labels='kmeans',
        random_state=RANDOM_STATE
    )
    sc.fit(adjacency)
    elapsed = time.perf_counter() - start
    return elapsed

def benchmark_gsc_network(adjacency, n_clusters, t=5, alpha=0.5):
    """
    Benchmark Generalized Spectral Clustering on network (precomputed adjacency).
    The measure is passed as (callable, kwargs) tuple - the pipeline resolves it.
    """
    start = time.perf_counter()
    sc = cluster.SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        laplacian_method='norm',
        standard=False,           # No symmetrization
        measure=(teleporting_undirected_measure, {'alpha': alpha, 't': t}),
        assign_labels='kmeans',
        random_state=RANDOM_STATE
    )
    sc.fit(adjacency)
    elapsed = time.perf_counter() - start
    return elapsed

# =============================================================================
# Main Experiments
# =============================================================================

def run_pointcloud_comparison():
    """
    Compare SC vs GSC complexity on point clouds.
    Both should exhibit O(N² log N) complexity.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Point Cloud Complexity (SC vs GSC)")
    print("=" * 70)
    
    results = []
    
    for N in N_VALUES_POINTCLOUD:
        print(f"\nN = {N:,}...")
        
        # Generate data once
        X, y = generate_point_cloud(N, D_VALUE, K_CLUSTERS)
        n_neighbors = int(np.ceil(np.log(N)))
        
        # Benchmark SC
        sc_times = []
        for i in range(N_ITERATIONS):
            t = benchmark_sc_pointcloud(X, K_CLUSTERS, n_neighbors)
            sc_times.append(t)
            print(f"  SC iteration {i+1}: {t:.2f}s")
        
        # Benchmark GSC
        gsc_times = []
        for i in range(N_ITERATIONS):
            t = benchmark_gsc_pointcloud(X, K_CLUSTERS, n_neighbors, t=5, alpha=0.5)
            gsc_times.append(t)
            print(f"  GSC iteration {i+1}: {t:.2f}s")
        
        results.append({
            'N': N,
            'D': D_VALUE,
            'k': n_neighbors,
            'SC_mean': np.mean(sc_times),
            'SC_std': np.std(sc_times),
            'GSC_mean': np.mean(gsc_times),
            'GSC_std': np.std(gsc_times),
        })
    
    df = pd.DataFrame(results)
    df['logN'] = np.log(df['N'])
    df['N2logN'] = df['N']**2 * df['logN']
    
    # Save results
    df.to_csv(f"{RESULTS_PATH}/pointcloud_sc_vs_gsc.csv", index=False)
    print(f"\n✓ Results saved to {RESULTS_PATH}/pointcloud_sc_vs_gsc.csv")
    
    return df

def run_network_experiment():
    """
    Benchmark complexity on networks with given adjacency matrix.
    Complexity: O(N × nnz × k) ≈ O(N² log N) for sparse networks.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Network Complexity (Given Adjacency)")
    print("=" * 70)
    
    results = []
    
    for N in N_VALUES_NETWORK:
        print(f"\nN = {N:,}...")
        
        # Generate sparse network
        adjacency, y = generate_sparse_network(N, K_CLUSTERS)
        nnz = adjacency.nnz
        print(f"  Edges: {nnz:,} (avg degree: {nnz/N:.1f})")
        
        # Benchmark SC
        sc_times = []
        for i in range(N_ITERATIONS):
            t = benchmark_sc_network(adjacency, K_CLUSTERS)
            sc_times.append(t)
            print(f"  SC iteration {i+1}: {t:.3f}s")
        
        # Benchmark GSC
        gsc_times = []
        for i in range(N_ITERATIONS):
            t = benchmark_gsc_network(adjacency, K_CLUSTERS, t=5, alpha=0.5)
            gsc_times.append(t)
            print(f"  GSC iteration {i+1}: {t:.3f}s")
        
        results.append({
            'N': N,
            'nnz': nnz,
            'avg_degree': nnz / N,
            'SC_mean': np.mean(sc_times),
            'SC_std': np.std(sc_times),
            'GSC_mean': np.mean(gsc_times),
            'GSC_std': np.std(gsc_times),
        })
    
    df = pd.DataFrame(results)
    df['logN'] = np.log(df['N'])
    df['N_nnz'] = df['N'] * df['nnz']  # O(N × nnz) term
    
    # Save results
    df.to_csv(f"{RESULTS_PATH}/network_complexity.csv", index=False)
    print(f"\n✓ Results saved to {RESULTS_PATH}/network_complexity.csv")
    
    return df

# =============================================================================
# Analysis and Plotting
# =============================================================================

def analyze_and_plot(df_pointcloud, df_network):
    """Generate publication-quality plots for complexity analysis."""
    
    print("\n" + "=" * 70)
    print("COMPLEXITY ANALYSIS")
    print("=" * 70)
    
    # Analyze point cloud results
    print("\n1. Point Cloud Complexity (SC vs GSC):")
    print("-" * 50)
    
    df = df_pointcloud
    for method in ['SC', 'GSC']:
        r2 = linregress(df['N2logN'], df[f'{method}_mean'])[2] ** 2
        print(f"  {method}: R²(T vs N² log N) = {r2:.4f}")
    
    # Analyze network results
    print("\n2. Network Complexity:")
    print("-" * 50)
    
    df = df_network
    for method in ['SC', 'GSC']:
        r2_n_nnz = linregress(df['N_nnz'], df[f'{method}_mean'])[2] ** 2
        print(f"  {method}: R²(T vs N×nnz) = {r2_n_nnz:.4f}")
    
    # ==========================================================================
    # Figure 1: Point Cloud - SC vs GSC
    # ==========================================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4.5))
    
    colors = {'SC': '#1f77b4', 'GSC': '#ff7f0e'}
    
    # Left: log-log time comparison
    ax = axes1[0]
    for method in ['SC', 'GSC']:
        ax.errorbar(df_pointcloud['N'], df_pointcloud[f'{method}_mean'],
                   yerr=df_pointcloud[f'{method}_std'],
                   fmt='o-', color=colors[method], markersize=7, linewidth=2,
                   label=method, capsize=3, markeredgecolor='white', markeredgewidth=0.5)
    
    # Reference line
    N_ref = np.linspace(500, 20000, 100)
    ref_point = df_pointcloud[df_pointcloud['N'] == 5000]['SC_mean'].values[0]
    ref_line = (N_ref**2 * np.log(N_ref)) / (5000**2 * np.log(5000)) * ref_point
    ax.loglog(N_ref, ref_line, 'k--', linewidth=2, alpha=0.6, label=r'$\mathcal{O}(N^2 \log N)$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of samples $N$')
    ax.set_ylabel('Computation time (seconds)')
    ax.set_title('(a) Point cloud: SC vs GSC')
    ax.legend(loc='upper left')
    
    # Right: Normalized time
    ax = axes1[1]
    for method in ['SC', 'GSC']:
        normalized = df_pointcloud[f'{method}_mean'] / df_pointcloud['N2logN'] * 1e9
        ax.plot(df_pointcloud['N'], normalized, 'o-', color=colors[method],
               markersize=7, linewidth=2, label=method,
               markeredgecolor='white', markeredgewidth=0.5)
    
    ax.set_xscale('log')
    ax.set_xlabel('Number of samples $N$')
    ax.set_ylabel(r'$T / (N^2 \log N)$ (nanoseconds)')
    ax.set_title(r'(b) Normalized time (constant for $\mathcal{O}(N^2 \log N)$)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    fig1.savefig(f"{RESULTS_PATH}/complexity_pointcloud_comparison.pdf", bbox_inches='tight')
    fig1.savefig(f"{RESULTS_PATH}/complexity_pointcloud_comparison.png", bbox_inches='tight', dpi=300)
    print(f"\n✓ Saved: complexity_pointcloud_comparison.pdf/png")
    
    # ==========================================================================
    # Figure 2: Network Complexity
    # ==========================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: Time vs N
    ax = axes2[0]
    for method in ['SC', 'GSC']:
        ax.errorbar(df_network['N'], df_network[f'{method}_mean'],
                   yerr=df_network[f'{method}_std'],
                   fmt='o-', color=colors[method], markersize=7, linewidth=2,
                   label=method, capsize=3, markeredgecolor='white', markeredgewidth=0.5)
    
    # Reference line for O(N × nnz) ≈ O(N² log N) for sparse graphs
    N_ref = np.linspace(500, 10000, 100)
    ref_point = df_network[df_network['N'] == 2000]['SC_mean'].values[0]
    ref_N2logN = (N_ref**2 * np.log(N_ref)) / (2000**2 * np.log(2000)) * ref_point
    ax.loglog(N_ref, ref_N2logN, 'k--', linewidth=2, alpha=0.6, label=r'$\mathcal{O}(N^2 \log N)$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of nodes $N$')
    ax.set_ylabel('Computation time (seconds)')
    ax.set_title('(a) Network: SC vs GSC')
    ax.legend(loc='upper left')
    
    # Right: Time vs N × nnz
    ax = axes2[1]
    for method in ['SC', 'GSC']:
        ax.plot(df_network['N_nnz'], df_network[f'{method}_mean'],
               'o-', color=colors[method], markersize=7, linewidth=2,
               label=method, markeredgecolor='white', markeredgewidth=0.5)
    
    # Linear fit line
    slope, intercept, _, _, _ = linregress(df_network['N_nnz'], df_network['SC_mean'])
    x_fit = np.linspace(df_network['N_nnz'].min(), df_network['N_nnz'].max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, 'k--', linewidth=2, alpha=0.6, label='Linear fit')
    
    ax.set_xlabel(r'$N \times \mathrm{nnz}$')
    ax.set_ylabel('Computation time (seconds)')
    ax.set_title(r'(b) Time vs $N \times \mathrm{nnz}$ (ARPACK complexity)')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    fig2.savefig(f"{RESULTS_PATH}/complexity_network.pdf", bbox_inches='tight')
    fig2.savefig(f"{RESULTS_PATH}/complexity_network.png", bbox_inches='tight', dpi=300)
    print(f"✓ Saved: complexity_network.pdf/png")
    
    # ==========================================================================
    # Figure 3: Combined Summary Figure
    # ==========================================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: Point cloud
    ax = axes3[0]
    for method in ['SC', 'GSC']:
        ax.errorbar(df_pointcloud['N'], df_pointcloud[f'{method}_mean'],
                   yerr=df_pointcloud[f'{method}_std'],
                   fmt='o-', color=colors[method], markersize=7, linewidth=2,
                   label=method, capsize=3, markeredgecolor='white', markeredgewidth=0.5)
    
    N_ref = np.linspace(500, 20000, 100)
    ref_point = df_pointcloud[df_pointcloud['N'] == 5000]['SC_mean'].values[0]
    ref_line = (N_ref**2 * np.log(N_ref)) / (5000**2 * np.log(5000)) * ref_point
    ax.loglog(N_ref, ref_line, 'k--', linewidth=2, alpha=0.6, label=r'$\mathcal{O}(N^2 \log N)$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of samples $N$')
    ax.set_ylabel('Computation time (seconds)')
    ax.set_title('(a) Point clouds ($k$-NN graph)')
    ax.legend(loc='upper left')
    
    # Right: Network
    ax = axes3[1]
    for method in ['SC', 'GSC']:
        ax.errorbar(df_network['N'], df_network[f'{method}_mean'],
                   yerr=df_network[f'{method}_std'],
                   fmt='o-', color=colors[method], markersize=7, linewidth=2,
                   label=method, capsize=3, markeredgecolor='white', markeredgewidth=0.5)
    
    N_ref = np.linspace(500, 10000, 100)
    ref_point = df_network[df_network['N'] == 2000]['SC_mean'].values[0]
    ref_line = (N_ref**2 * np.log(N_ref)) / (2000**2 * np.log(2000)) * ref_point
    ax.loglog(N_ref, ref_line, 'k--', linewidth=2, alpha=0.6, label=r'$\mathcal{O}(N^2 \log N)$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of nodes $N$')
    ax.set_ylabel('Computation time (seconds)')
    ax.set_title('(b) Networks (given adjacency)')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    fig3.savefig(f"{RESULTS_PATH}/complexity_summary.pdf", bbox_inches='tight')
    fig3.savefig(f"{RESULTS_PATH}/complexity_summary.png", bbox_inches='tight', dpi=300)
    print(f"✓ Saved: complexity_summary.pdf/png")
    
    plt.close('all')
    
    return fig1, fig2, fig3

def print_latex_summary(df_pointcloud, df_network):
    """Print LaTeX-formatted summary tables."""
    
    print("\n" + "=" * 70)
    print("LATEX TABLES")
    print("=" * 70)
    
    # R² summary
    print(r"""
\begin{table}[h]
\centering
\caption{Coefficient of determination ($R^2$) for $T \propto N^2 \log N$ complexity model.}
\label{tab:complexity_r2}
\begin{tabular}{l|cc}
\toprule
Data Type & SC & GSC \\
\midrule""")
    
    # Point cloud
    r2_sc = linregress(df_pointcloud['N2logN'], df_pointcloud['SC_mean'])[2] ** 2
    r2_gsc = linregress(df_pointcloud['N2logN'], df_pointcloud['GSC_mean'])[2] ** 2
    print(f"Point clouds ($D={D_VALUE}$) & {r2_sc:.4f} & {r2_gsc:.4f} \\\\")
    
    # Network
    r2_sc = linregress(df_network['N_nnz'], df_network['SC_mean'])[2] ** 2
    r2_gsc = linregress(df_network['N_nnz'], df_network['GSC_mean'])[2] ** 2
    print(f"Networks (sparse) & {r2_sc:.4f} & {r2_gsc:.4f} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

# =============================================================================
# Main
# =============================================================================

def main():
    """Run the complete complexity analysis."""
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COMPLEXITY ANALYSIS: GSC FRAMEWORK")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Point cloud N values: {N_VALUES_POINTCLOUD}")
    print(f"  Point cloud dimension: {D_VALUE}")
    print(f"  Network N values: {N_VALUES_NETWORK}")
    print(f"  Number of clusters: {K_CLUSTERS}")
    print(f"  Iterations per measurement: {N_ITERATIONS}")
    
    # Run experiments
    df_pointcloud = run_pointcloud_comparison()
    df_network = run_network_experiment()
    
    # Analyze and plot
    analyze_and_plot(df_pointcloud, df_network)
    
    # Print LaTeX tables
    print_latex_summary(df_pointcloud, df_network)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_PATH}/")
    print("  - pointcloud_sc_vs_gsc.csv")
    print("  - network_complexity.csv")
    print("  - complexity_pointcloud_comparison.pdf/png")
    print("  - complexity_network.pdf/png")
    print("  - complexity_summary.pdf/png")

if __name__ == '__main__':
    main()
