"""Generate a 3-Gaussian dataset with one dense cluster and two sparse clusters."""

from pathlib import Path

import numpy as np


def degree_imbalanced_gaussians(
	cluster_sizes=(300, 300, 300),
	centers=((0.0, 0.0), (4.0, 0.0), (2.0, 3.5)),
	dense_std=0.35,
	sparse_std=1.0,
	seed=42,
):
	"""Generate three Gaussian blobs with a local-density imbalance.

	The first cluster is compact (dense), while the two others are more spread out
	(sparse). With equal sample counts, a kNN graph built on this dataset will tend
	to give higher local degrees around the dense cluster than around the sparse ones.

	Parameters
	----------
	cluster_sizes : tuple of int, optional
		Number of samples in each cluster.
	centers : tuple of tuple of float, optional
		2D centers of the three Gaussian components.
	dense_std : float, optional
		Standard deviation of the first, dense Gaussian.
	sparse_std : float, optional
		Standard deviation of the second and third, sparse Gaussians.
	seed : int, optional
		Random seed.

	Returns
	-------
	X : numpy.ndarray
		Array of shape (n_samples, 2).
	labels : numpy.ndarray
		Ground-truth labels of shape (n_samples,).
	"""
	if len(cluster_sizes) != 3:
		raise ValueError("cluster_sizes must contain exactly 3 values")

	if len(centers) != 3:
		raise ValueError("centers must contain exactly 3 2D centers")

	rng = np.random.default_rng(seed)

	stds = (dense_std, sparse_std, sparse_std)
	cluster_data = []
	labels = []

	for cluster_id, (size, center, std) in enumerate(zip(cluster_sizes, centers, stds)):
		if size <= 0:
			raise ValueError("All cluster sizes must be positive")
		if std <= 0:
			raise ValueError("All standard deviations must be positive")

		covariance = np.array([[std**2, 0.0], [0.0, std**2]])
		points = rng.multivariate_normal(mean=center, cov=covariance, size=size)
		cluster_data.append(points)
		labels.extend([cluster_id] * size)

	X = np.vstack(cluster_data)
	y = np.asarray(labels)

	return X, y


def _save_npz_dataset(output_path: str | Path, X: np.ndarray, labels: np.ndarray) -> None:
	"""Save the generated dataset to a compact NumPy archive."""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez(output_path, X=X, labels=labels)


def main():
	import argparse

	parser = argparse.ArgumentParser(
		description="Generate 3 Gaussian clusters with degree imbalance"
	)
	parser.add_argument(
		"--cluster-sizes",
		type=int,
		nargs=3,
		default=(300, 300, 300),
		help="Sizes of the 3 clusters",
	)
	parser.add_argument(
		"--dense-std",
		type=float,
		default=0.35,
		help="Standard deviation of the dense cluster",
	)
	parser.add_argument(
		"--sparse-std",
		type=float,
		default=1.0,
		help="Standard deviation of the sparse clusters",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Optional .npz file path to save the dataset",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display a scatter plot of the generated dataset",
	)

	args = parser.parse_args()

	X, labels = degree_imbalanced_gaussians(
		cluster_sizes=tuple(args.cluster_sizes),
		dense_std=args.dense_std,
		sparse_std=args.sparse_std,
		seed=args.seed,
	)

	print(f"Generated dataset with shape: {X.shape}")
	print(f"Cluster sizes: {tuple(args.cluster_sizes)}")
	print(
		"Cluster stds: "
		f"dense={args.dense_std:.3f}, sparse={args.sparse_std:.3f}, {args.sparse_std:.3f}"
	)

	if args.output:
		_save_npz_dataset(args.output, X, labels)
		print(f"Saved dataset to: {args.output}")

	if args.show:
		import matplotlib.pyplot as plt

		colors = np.array(["#072AC8", "#27A727", "#FF6347"])
		plt.figure(figsize=(6, 6))
		plt.scatter(X[:, 0], X[:, 1], c=colors[labels], s=10, alpha=0.8)
		plt.gca().set_aspect("equal", adjustable="box")
		plt.title("Degree-Imbalanced Gaussian Dataset")
		plt.tight_layout()
		plt.show()


if __name__ == "__main__":
	main()
