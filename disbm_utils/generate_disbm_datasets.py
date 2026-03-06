"""
Directed SBM dataset generation and utilities.

Given a meta-graph probability matrix $A in [0,1]^{K \times K}$, we generate a DSBM graph with $n$ nodes and $K$ communities as follows:

$P_{ij} = A_{g_i g_j}$ where $g_i$ is the community assignment of node $i$. Then we sample edges according to $P$.

Here, $A$ controls the intra- and inter-community connection probabilities, $A$ is generally not symmetric.
"""

import networkx as nx
import numpy as np


def directed_sbm(block_sizes: list, P: np.ndarray, seed: int = 42):
    """
    :param block_sizes: List of sizes for each block
    :param P: Probability matrix for edges between blocks
    :return: Directed SBM graph and ground truth labels
    """
    G = nx.stochastic_block_model(
        sizes=block_sizes, p=P, directed=True, selfloops=False, seed=seed
    )
    y = []
    for i, size in enumerate(block_sizes):
        y.extend([i] * size)
    return nx.adjacency_matrix(G), y