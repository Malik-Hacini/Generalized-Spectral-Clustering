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


def core_periphery_disbm(
    block_sizes: list,
    p_core: float,
    p_periphery: float,
    p_core_periphery: float,
    p_periphery_core: float,
    seed: int = 42,
):
    """
    Generate a core-periphery DSBM graph with specified probabilities for core-core, periphery-periphery, and core-periphery connections.

    :param block_sizes: List of sizes, the first block is the core, the others are periphery blocks
    :param p_core: Probability of edges within the core
    :param p_periphery: Probability of edges within the periphery
    :param p_core_periphery: Probability of edges from core to periphery
    :param p_periphery_core: Probability of edges from periphery to core

    Generally, we expect p_core > p_core_periphery >> p_periphery > p_periphery_core for a strong core-periphery structure.

    :return: Directed SBM graph and ground truth labels
    """

    K = len(block_sizes)
    P = np.full((K, K), p_periphery)  # Start with periphery probability
    P[0, 0] = p_core  # Core-core
    P[0, 1:] = p_core_periphery  # Core to periphery
    P[1:, 0] = p_periphery_core  # Periphery to core
    for i in range(1, K):
        P[i, i] = p_periphery  # Periphery-periphery
    return directed_sbm(block_sizes, P, seed)


def chain_sbm(
    block_sizes: list,
    p_intra: float,
    p_forward: float,
    p_backward: float,
):
    """
    Generate a chain-structured DSBM graph where each block is connected to the next with a forward probability and to the previous with a backward probability.
    :param block_sizes: List of sizes for each block
    :param p_intra: Probability of edges within each block
    :param p_forward: Probability of edges from block i to block i+1
    :param p_backward: Probability of edges from block i to block i-1

    Generally, we expect p_intra > p_forward >> p_backward for a strong chain structure.

    :return: Directed SBM graph and ground truth labels
    """

    K = len(block_sizes)
    P = np.full((K, K), p_backward)  # Start with backward probability
    np.fill_diagonal(P, p_intra)  # Intra-block
    for i in range(K - 1):
        P[i, i + 1] = p_forward  # Forward connection
    return directed_sbm(block_sizes, P)


def degree_imabalance_sbm(
    block_sizes: list,
    p_intra: float,
    p_high: float,
    p_low: float,
    seed: int = 42,
):
    """
    Generate a DSBM graph with degree imbalance between two groups of blocks.

    :param block_sizes: List of sizes for each block, we assume the first half are "high-degree" and the second half are "low-degree"
    :param p_intra: Probability of edges within each block
    :param p_high: Probability of edges from high-degree blocks to any block
    :param p_low: Probability of edges from low-degree blocks to any block

    Generally, we expect p_high >> p_low for a strong degree imbalance.

    :return: Directed SBM graph and ground truth labels
    """

    K = len(block_sizes)
    P = np.full((K, K), p_low)  # Start with low probability
    np.fill_diagonal(P, p_intra)  # Intra-block
    P[0, :] = p_high
    return directed_sbm(block_sizes, P, seed)