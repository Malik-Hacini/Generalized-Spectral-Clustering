"""Runtime benchmark vs dataset size for SC-UN and GSC-UN variants.

This script builds synthetic Gaussian point-cloud datasets with increasing size
and compares runtime of:
1) SC-UN (standard unnormalized spectral clustering)
n_seeds = 5
3) GSC-UN without tuning (fixed t, alpha)
"""

# A small one-off warm-up dataset to absorb first-call overhead
# (imports, BLAS thread pool init, sklearn internals).
warmup_n_samples = 450
warmup_seed = 2026

if __package__ is None or __package__ == "":
	from common import *
else:
	from experiments.common import *

from utils.file_manager import save_dataset


def _fmt_float(value: float) -> str:
	return f"{value:.4f}".replace(".", "p")


def _sample_gaussian_mixture(
	n_samples: int,
	centers: np.ndarray,
	std: float,
	seed: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Sample balanced Gaussian clusters with total size n_samples."""
	rng = np.random.default_rng(seed)
	n_clusters = centers.shape[0]

	base = n_samples // n_clusters
	rem = n_samples % n_clusters
	sizes = [base + (1 if i < rem else 0) for i in range(n_clusters)]

	X_parts = []
	y_parts = []
	for k, nk in enumerate(sizes):
		if nk == 0:
			continue
		Xk = rng.normal(loc=centers[k], scale=std, size=(nk, centers.shape[1]))
		yk = np.full(nk, k, dtype=int)
		X_parts.append(Xk)
		y_parts.append(yk)

	X = np.vstack(X_parts)
	y = np.concatenate(y_parts)

	perm = rng.permutation(len(y))
	return X[perm], y[perm]


"""
Basic experiment config:
"""
save_path = project_path("results")
experiment_name = "benchmark_runtimes_size"
mode = "grid_search"
metrics = ("ami", "ch")
n_jobs = -1
verbose = True


"""
Synthetic Gaussian datasets with increasing size.
"""
dataset_sizes = [600, 1200, 2400, 3600, 4800, 6000, 7200, 9600]
n_seeds = 10
centers = np.array([[0.0, 0.0], [3.0, 0.2], [1.5, 3.0]], dtype=float)
std = 0.45

datasets_path = project_path("datasets/runtimes")
Path(datasets_path).mkdir(parents=True, exist_ok=True)

print("Generating size-scaling Gaussian datasets...")
dataset_names = []
for n_samples in dataset_sizes:
	for seed in range(n_seeds):
		dataset_name = (
			f"gauss_runtime_n{n_samples}"
			f"_k{centers.shape[0]}"
			f"_std{_fmt_float(std)}"
			f"_seed{seed}"
		)
		dataset_dir = Path(datasets_path) / dataset_name
		train_dir = dataset_dir / "train"

		if not train_dir.exists():
			X, labels = _sample_gaussian_mixture(
				n_samples=n_samples,
				centers=centers,
				std=std,
				seed=seed,
			)
			save_dataset(
				data=X,
				labels=labels,
				path=datasets_path,
				name=dataset_name,
				feature_cols=["x", "y"],
				label_col="labels",
			)
			print(f"  Created: {dataset_name}")

		dataset_names.append(dataset_name)

print(f"Total datasets: {len(dataset_names)}")

# Warm-up dataset is generated separately and not included in final benchmark curves.
warmup_dataset_name = (
	f"gauss_runtime_warmup_n{warmup_n_samples}"
	f"_k{centers.shape[0]}"
	f"_std{_fmt_float(std)}"
	f"_seed{warmup_seed}"
)
warmup_dataset_dir = Path(datasets_path) / warmup_dataset_name
warmup_train_dir = warmup_dataset_dir / "train"
if not warmup_train_dir.exists():
	X_warm, y_warm = _sample_gaussian_mixture(
		n_samples=warmup_n_samples,
		centers=centers,
		std=std,
		seed=warmup_seed,
	)
	save_dataset(
		data=X_warm,
		labels=y_warm,
		path=datasets_path,
		name=warmup_dataset_name,
		feature_cols=["x", "y"],
		label_col="labels",
	)
	print(f"  Created warm-up dataset: {warmup_dataset_name}")


"""
Methods configuration:
"""
method_specs = [
	("spectral", "SC-UN"),
	("spectral", "GSC-UN"),
	("spectral", "GSC-UN-NoTune"),
]


"""
Parameters configuration:
"""
default_params = {
	"n_neighbors": 10,
	"random_state": 42,
	"affinity": "nearest_neighbors",
	"n_it": 1,
	"assign_labels": "kmeans",
}

method_params = [
	(
		"SC-UN",
		{
			"laplacian_method": "unnorm",
			"standard": True,
			"measure": None,
		},
	),
	(
		"GSC-UN",
		{
			"laplacian_method": "unnorm",
			"measure": (
				teleporting_undirected_measure,
				{"alpha": np.arange(0.0, 1.5, 0.1), "t": range(0, 21, 2)},
			),
		},
	),
	(
		"GSC-UN-NoTune",
		{
			"laplacian_method": "unnorm",
			"measure": (
				teleporting_undirected_measure,
				{"alpha": 0.4, "t": 10},
			),
		},
	),
]

dataset_params = []
method_dataset_params = []


"""
Do not edit below unless you really want to!
"""
config = ExperimentConfig(
	default_params=default_params,
	dataset_params=dataset_params,
	method_params=method_params,
	method_dataset_params=method_dataset_params,
)

if __name__ == "__main__":
	print("Running one warm-up pass to reduce first-run timing bias...")
	_ = experiment(
		experiment_name=f"{experiment_name}_warmup",
		dataset_names=[warmup_dataset_name],
		method_specs=method_specs,
		config=config,
		load_path=datasets_path,
		save_path=save_path,
		mode=mode,
		metrics=metrics,
		n_jobs=1,
		verbose=False,
	)

	start = time.time()
	_ = experiment(
		experiment_name=experiment_name,
		dataset_names=dataset_names,
		method_specs=method_specs,
		config=config,
		load_path=datasets_path,
		save_path=save_path,
		mode=mode,
		metrics=metrics,
		n_jobs=n_jobs,
		verbose=verbose,
	)
	end = time.time()
	print(f"Experiment completed in {end - start:.2f} seconds.")
