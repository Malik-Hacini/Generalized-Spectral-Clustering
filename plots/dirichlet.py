import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import scipy.sparse as sp

directed = True

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
k = 10  # number of neighbors (increased to ensure full connectivity)
A_knn = kneighbors_graph(data, n_neighbors=k, mode='connectivity', include_self=False)
# Convert sparse matrix to COO format and construct graph from edges
A_knn_coo = sp.coo_matrix(A_knn)
edges = list(zip(A_knn_coo.row, A_knn_coo.col))
G_knn = nx.DiGraph()
G_knn.add_nodes_from(range(A_knn.shape[0]))
G_knn.add_edges_from(edges)

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
# Convert sparse matrix to COO format and construct graph from edges
A_knn_asym_coo = sp.coo_matrix(A_knn_asym)
edges = list(zip(A_knn_asym_coo.row, A_knn_asym_coo.col))
G_knn_asym = nx.DiGraph()
G_knn_asym.add_nodes_from(range(A_knn_asym.shape[0]))
G_knn_asym.add_edges_from(edges)

# Compute ergodic law (stationary distribution) from the transition matrix
A_asym = sp.csr_matrix(A_knn_asym).toarray()

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

# Check reciprocity (how many edges are bidirectional)
reciprocal_edges = sum(1 for u, v in G_knn_asym.edges() if G_knn_asym.has_edge(v, u))
print(f"Reciprocal edges: {reciprocal_edges}/{G_knn_asym.number_of_edges()} ({100*reciprocal_edges/G_knn_asym.number_of_edges():.1f}%)")
print(f"Using {'directed' if directed else 'undirected'} graph for ergodic law computation")
if use_zero_mask:
    zero_mask = np.abs(node_colors) < 1e-10
    print(f"Points with zero distribution: {np.sum(zero_mask)}/{len(node_colors)}")


# Create two functions with the same Dirichlet energy
# Function 1: Use true cluster labels (assuming 3 clusters)
# Function 2: Mix first two clusters randomly, keep third cluster well-separated

# First, let's identify which points belong to which cluster
# Assuming we have 3 clusters with equal numbers of points
n_clusters = 3
cluster_size = len(data) // n_clusters
true_labels = np.repeat(np.arange(n_clusters), cluster_size)

# Create function 1: based on true cluster membership
# Assign distinct values to each cluster
cluster_values = {0: -1.0, 1: 0.0, 2: 1.0}
f1_true = np.array([cluster_values[int(label)] for label in true_labels])

# Compute Dirichlet energy for f1
energy_f1 = dirichlet_energy(f1_true, P, stationary_dist)
print(f"Dirichlet energy for true labels: {energy_f1:.6f}")

# Create function 2: mix first two clusters randomly, keep third intact
# Strategy: randomly assign labels from clusters 0 and 1 to the first two clusters
# Keep cluster 2 with its original label

np.random.seed(42)
f2_mixed_labels = true_labels.copy()

# For clusters 0 and 1 (first 2*cluster_size points), randomly shuffle labels between 0 and 1
mixed_indices = np.arange(2 * cluster_size)
# Randomly assign labels 0 or 1 to these points
f2_mixed_labels[mixed_indices] = np.random.choice([0, 1], size=len(mixed_indices))

# Convert labels to function values
f2_mixed = np.array([cluster_values[int(label)] for label in f2_mixed_labels])

# Compute initial energy
energy_f2_initial = dirichlet_energy(f2_mixed, P, stationary_dist)
print(f"Initial Dirichlet energy for mixed version: {energy_f2_initial:.6f}")

# Fine-tune to match energies by scaling
# Energy scales quadratically with function values
if energy_f2_initial > 0:
    f2_mixed_scaled = f2_mixed
    energy_f2_scaled = dirichlet_energy(f2_mixed_scaled, P, stationary_dist)
    print(f"Scaled Dirichlet energy for mixed version: {energy_f2_scaled:.6f}")
else:
    f2_mixed_scaled = f2_mixed
    energy_f2_scaled = energy_f2_initial

# Define discrete colors for clusters
colors_true = ['#072AC8', '#FFBF46', '#FF1F2E']  # Blue, Yellow, Red
colors_mixed = ['#072AC8', '#FFBF46', '#FF1F2E']  # Blue, Yellow, Red for mixed case

def plot_function(f_values, filename, use_colors, directed=True):
    """Helper function to plot a single function"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot edges with arrows (directed graph)
    for edge in G_knn_asym.edges():
        i, j = edge
        if directed:
            ax.annotate('', xy=(data[j, 0], data[j, 1]), xytext=(data[i, 0], data[i, 1]),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3,
                                       lw=0.8, shrinkA=5, shrinkB=5),
                        zorder=0)
        else:
            ax.plot([data[i, 0], data[j, 0]],
                   [data[i, 1], data[j, 1]],
                   'gray', alpha=0.2, linewidth=0.5, zorder=0)

    # Plot nodes with discrete colors (no colormap)
    # Map function values to cluster indices
    unique_vals = np.sort(np.unique(f_values))
    for cluster_idx, val in enumerate(unique_vals):
        mask = np.isclose(f_values, val, atol=0.01)
        ax.scatter(data[mask, 0], data[mask, 1],
                  c=use_colors[cluster_idx % len(use_colors)],
                  s=100, alpha=1, linewidth=0.5, zorder=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf")

# Plot 1: True labels
plot_function(f1_true, 'dirichlet_true_labels', colors_true, directed=True)

# Plot 2: Mixed version (use only blue and yellow for first two clusters, red for third)
plot_function(f2_mixed_scaled, 'dirichlet_mixed_labels', colors_true, directed=True)

print(f"\nEnergy difference: {abs(energy_f1 - energy_f2_scaled):.8f}")
print(f"Relative energy difference: {abs(energy_f1 - energy_f2_scaled) / energy_f1 * 100:.4f}%")