"""
Graph Calinski-Harabasz index based on random walk (diffusion) distance.

Defines a clustering quality metric for graphs using a distance derived from
the transition matrix P = D^{-1}A (random walk matrix). Supports general
polynomial filters g(P) = sum_k a_k P^k for multi-scale distance computation.

References
----------
Coifman, R. R. & Lafon, S. (2006). "Diffusion maps."
    Applied and Computational Harmonic Analysis, 21(1), 5-30.

Calinski, T. & Harabasz, J. (1974). "A dendrite method for cluster analysis."
    Communications in Statistics, 3(1), 1-27.

See derivation.md in this folder for the full mathematical derivation.
"""

from .graph_ch import graph_calinski_harabasz

__all__ = ["graph_calinski_harabasz"]
