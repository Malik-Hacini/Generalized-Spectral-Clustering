import numpy as np
import scipy.sparse as sp

def compute_signal_ratio(F: np.ndarray) -> tuple[float, float]:
    """
    Computes the theoretical symmetric and directed signal ratios
    from the meta-graph probability matrix F.
    
    Returns:
        sym_ratio: Ratio of diagonal (intra) to symmetric off-diagonal (inter) probabilities.
        dir_ratio: Ratio of flow imbalance between off-diagonal blocks.
    """
    K = F.shape[0]
    
    # Symmetric signal
    S = 0.5 * (F + F.T)
    diag_mean = np.trace(S) / K
    off_diag_mask = ~np.eye(K, dtype=bool)
    off_diag_mean = S[off_diag_mask].mean()
    
    sym_ratio = diag_mean / off_diag_mean if off_diag_mean > 0 else np.inf
    
    # Directed signal (flow imbalance)
    A = F - F.T
    # Average positive flow
    pos_flow_mean = A[A > 0].mean() if np.any(A > 0) else 0.0
    
    # Just return the raw flow strength
    dir_flow = pos_flow_mean
    
    return sym_ratio, dir_flow

def main():
    F_sym = np.array([
        [0.9, 0.3, 0.3],
        [0.3, 0.9, 0.3],
        [0.3, 0.3, 0.9]
    ])
    
    F_cyclic = np.array([
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 1.0],
        [1.0, 0.0, 0.5]
    ])
    
    gammas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print(f"{'gamma':<6} | {'sym_ratio':<10} | {'dir_flow':<10}")
    print("-" * 33)
    
    for gamma in gammas:
        F = (1.0 - gamma) * F_sym + gamma * F_cyclic
        sym, dir_f = compute_signal_ratio(F)
        print(f"{gamma:<6.1f} | {sym:<10.3f} | {dir_f:<10.3f}")

if __name__ == "__main__":
    main()
