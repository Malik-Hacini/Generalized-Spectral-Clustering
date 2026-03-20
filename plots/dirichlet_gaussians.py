
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import networkx as nx

def dirichlet_energy(f, P, stationary_dist):
    """
    Compute Dirichlet energy: sum_{i,j} pi(i) * P_{i,j} * [f(i) - f(j)]^2

    Parameters:
    -----------
    f : array-like, shape (n,)
        Function values on nodes
    P : array-like, shape (n, n)
        Transition matrix
    stationary_dist : array-like, shape (n,)
        Stationary distribution π(i)

    Returns:
    --------
    float : Dirichlet energy
    """
    n = len(f)
    energy = 0.0
    for i in range(n):
        for j in range(n):
            if P[i, j] > 0:
                energy += stationary_dist[i] * P[i, j] * (f[i] - f[j])**2
    return energy


directed = True

# Generate 3 clusters from Gaussian distributions
np.random.seed(31)

# Define cluster parameters (mean and covariance)
n_points_per_cluster = 50
cluster_params = [
    {'mean': [0, 0], 'cov': [[0.5, 0], [0, 0.5]]},      # Cluster 1: center
    {'mean': [3, 3], 'cov': [[0.4, 0.15], [0.15, 0.4]]},  # Cluster 2: upper right
    {'mean': [2, -4], 'cov': [[0.6, -0.2], [-0.2, 0.6]]}  # Cluster 3: lower
]

# Generate data
clusters = []
labels_true = []
for i, params in enumerate(cluster_params):
    cluster_data = np.random.multivariate_normal(
        params['mean'],
        params['cov'],
        n_points_per_cluster
    )
    clusters.append(cluster_data)
    labels_true.extend([i] * n_points_per_cluster)

# Combine all clusters
data = np.vstack(clusters)
labels_true = np.array(labels_true)

# Build k-NN graph
k = 10
A_knn = kneighbors_graph(data, n_neighbors=k, mode='connectivity', include_self=False)
G_knn = nx.from_scipy_sparse_array(A_knn, create_using=nx.DiGraph())

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Define colors for clusters
colors = ['#072AC8', '#FFBF46', '#FF1F2E']

# Plot data points colored by their true cluster
for i in range(3):
    cluster_mask = labels_true == i
    ax.scatter(data[cluster_mask, 0], data[cluster_mask, 1],
              c=colors[i], s=100, alpha=1, linewidth=0.5)

# Plot k-NN edges
pos = {i: data[i] for i in range(len(data))}
for edge in G_knn.edges():
    i, j = edge
    if not directed:
      ax.plot([data[i, 0], data[j, 0]],
            [data[i, 1], data[j, 1]],
            'gray', alpha=0.3, linewidth=0.8, zorder=0,
            )
    else:
      ax.annotate('', xy=(data[j, 0], data[j, 1]), xytext=(data[i, 0], data[i, 1]),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3,
                               lw=0.8, shrinkA=5, shrinkB=5),
                zorder=0)

# Remove axis labels, ticks, and frame
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
name = "figures/clustering"
if directed:
  name += "_directed"
plt.savefig(f"{name}.svg")
plt.show()

# Print some statistics
print(f"Total points: {len(data)}")
print(f"Points per cluster: {n_points_per_cluster}")
print(f"k-NN parameter: k={k}")
print(f"Total edges in graph: {G_knn.number_of_edges()}")
print(f"Average in-degree: {np.mean([d for n, d in G_knn.in_degree()]):.2f}")
print(f"Is weakly connected: {nx.is_weakly_connected(G_knn)}")


directed = True  # Set to False to use underlying undirected graph
use_zero_mask = False  # Set to False to use colormap for all points including zeros

# Build k-NN graph (ASYMMETRIC - not symmetrized)
k = 10
A_knn_asym = kneighbors_graph(data, n_neighbors=k, mode='connectivity', include_self=False)
G_knn_asym = nx.from_scipy_sparse_array(A_knn_asym, create_using=nx.DiGraph())

# Compute ergodic law (stationary distribution) from the transition matrix
A_asym = A_knn_asym.toarray()

if not directed:
    # Use underlying undirected graph by symmetrizing
    A_for_ergodic = A_asym + A_asym.T
else:
    # Use directed graph as-is
    A_for_ergodic = A_asym

# Compute row-normalized transition matrix P
out_degrees = A_for_ergodic.sum(axis=1)
out_degrees[out_degrees == 0] = 1  # Avoid division by zero
P = A_for_ergodic / out_degrees[:, np.newaxis]

# Find stationary distribution: left eigenvector of P with eigenvalue 1
eigenvalues, eigenvectors = np.linalg.eig(P.T)
# Find eigenvector corresponding to eigenvalue closest to 1
stationary_idx = np.argmin(np.abs(eigenvalues - 1))
stationary_dist = np.real(eigenvectors[:, stationary_idx])
stationary_dist = stationary_dist / stationary_dist.sum()  # Normalize to sum to 1

node_colors = stationary_dist

# Create the plot
fig, ax = plt.subplots(figsize=(9, 8))

# Create custom colormap from gray to red
from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list('gray_to_red', ['#A3D9FF', '#BF1363'])

if use_zero_mask:
    # Separate points with exactly 0 distribution from others
    zero_mask = np.abs(node_colors) < 1e-10  # Use small threshold for numerical precision
    nonzero_mask = ~zero_mask

    # Plot non-zero points with colormap
    if np.any(nonzero_mask):
        scatter = ax.scatter(data[nonzero_mask, 0], data[nonzero_mask, 1],
                            c=node_colors[nonzero_mask],
                            s=100, alpha=1, linewidth=0.5,
                            cmap=custom_cmap,
                            vmin=node_colors[nonzero_mask].min(),
                            vmax=node_colors[nonzero_mask].max(),
                          )

        # Add horizontal colorbar with reduced size and only extreme values
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.4, orientation='horizontal', pad=0.05, format='%.2f')
        cbar.set_label('Stationary Distribution', labelpad=10)
        # Set ticks to show only min and max values
        cbar.set_ticks([node_colors[nonzero_mask].min(), node_colors[nonzero_mask].max()])

    # Plot zero points in black
    if np.any(zero_mask):
        ax.scatter(data[zero_mask, 0], data[zero_mask, 1],
                  c='#353238', s=100, alpha=1, linewidth=0.5,
                   label='Zero distribution')
else:
    # Plot all points with colormap (no special handling for zeros)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=node_colors,
                        s=100, alpha=1, linewidth=0.5,
                        cmap=custom_cmap, vmin=min(node_colors), vmax=max(node_colors),
                        )

    # Add horizontal colorbar with reduced size and only extreme values
    cbar = plt.colorbar(scatter,
                        ax=ax,
                        shrink=0.4,
                        # orientation='horizontal',
                        pad=0.05,
                        format='%.2f')
    cbar.set_label('Stationary Distribution', labelpad=-10)
    # Set ticks to show only min and max values
    cbar.set_ticks([min(node_colors), max(node_colors)])

# Plot k-NN edges (with arrows if directed)
pos = {i: data[i] for i in range(len(data))}
for edge in G_knn_asym.edges():
    i, j = edge
    if directed:
        ax.annotate('', xy=(data[j, 0], data[j, 1]), xytext=(data[i, 0], data[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3,
                                   lw=0.8, shrinkA=5, shrinkB=5),
                    zorder=0)
    else:
        # For undirected visualization, plot edges without arrows
        ax.plot([data[i, 0], data[j, 0]],
               [data[i, 1], data[j, 1]],
               'gray', alpha=0.3, linewidth=0.8, zorder=0)

# Remove axis labels, ticks, and frame
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

filename = "figures/clustering_ergodic"
if not directed:
    filename += "_undirected"
plt.savefig(f"{filename}.pdf", bbox_inches='tight')
plt.show()

# Check reciprocity (how many edges are bidirectional)
reciprocal_edges = sum(1 for u, v in G_knn_asym.edges() if G_knn_asym.has_edge(v, u))
print(f"Reciprocal edges: {reciprocal_edges}/{G_knn_asym.number_of_edges()} ({100*reciprocal_edges/G_knn_asym.number_of_edges():.1f}%)")
print(f"Using {'directed' if directed else 'undirected'} graph for ergodic law computation")
if use_zero_mask:
    zero_mask = np.abs(node_colors) < 1e-10
    print(f"Points with zero distribution: {np.sum(zero_mask)}/{len(node_colors)}")
