"""
Clustering benchmark performed in the GSC paper.
"""

if __package__ is None or __package__ == "":
    from common import *
else:
    from experiments.common import *

from utils.file_manager import save_graph_dataset

"""
Basic experiment config:
"""
save_path = project_path("results")
experiment_name = "networks"
mode = "grid_search"  # Either "score", "grid_search" or "viz" when all datasets are 2D.
metrics = ("ami", "graph_ch")  # Valid metrics: "ami", "ari", "nmi", "ch"
n_jobs = (
    -1
)  # Number of parallel jobs (set to 1 for single-threaded execution, -1 to use all available cores)
verbose = True

"""
Datasets and methods configuration:
"""
load_path = project_path("datasets")
dataset_names = [
    "polblogs",
    "karate",
    "football",
    "email_eu_core",
    "wiki_vote",
    # "lead_lag"
    "polbooks"
]

"""
Synthetic directed-network datasets with fixed parameters for the paper/document.
"""
synthetic_network_specs = [
    {
        "name": "chain_sbm_fixed",
        "builder": chain_sbm,
        "params": {
            "block_sizes": [350, 350, 350],
            "p_intra": 0.12,
            "p_forward": 0.06,
            "p_backward": 0.01,
            "seed": 42,
        },
    },
    {
        "name": "core_periphery_disbm_fixed",
        "builder": core_periphery_disbm,
        "params": {
            "block_sizes": [350, 350, 350],
            "p_core": 0.14,
            "p_periphery": 0.02,
            "p_core_periphery": 0.12,
            "p_periphery_core": 0.01,
            "seed": 42,
        },
    },
]

for spec in synthetic_network_specs:
    dataset_name = spec["name"]
    dataset_dir = Path(load_path) / dataset_name
    graph_file = dataset_dir / "graph.npz"

    if not graph_file.exists():
        adjacency_matrix, labels = spec["builder"](**spec["params"])
        save_graph_dataset(
            adjacency_matrix=adjacency_matrix,
            labels=labels,
            path=load_path,
            name=dataset_name,
        )
        print(f"Created synthetic network dataset: {dataset_name}")

    dataset_names.append(dataset_name)

method_specs = [
    ("spectral", "SC-UN"),
    ("spectral", "SC-N"),
    ("dsc", "DSC+"),
    ("di_sim", "DI-SIM-R"),
    ("di_sim", "DI-SIM-L"),
    ("di_sim", "DI-SIM-C"),
    ("spectral", "GSC-N"),
    ("spectral", "GSC-UN"),
    ("spectral", "deg-GSC-N"),
    ("spectral", "deg-GSC-UN"),
    ("spectral", "uniform-GSC-N"),
    ("spectral", "uniform-GSC-UN"),
    ("spectral", "perron-GSC-N"),
    ("spectral", "perron-GSC-UN"),
    # (Internal  name, Display name ) - see utils.experiments_utils.clusterer
]

# Parameters
"""
Parameter hierarchy (lowest to highest precedence):
1. default_params : Base parameters for all experiments
2. General dataset parameters : [(dataset_name, params_dict), ...]
3. General method parameters: [(method_name, params_dict), ...]
4. Specific parameters for a method/dataset combination : [(method_name, [(dataset, params), ...]), ...]

Notes :
    - The number of clusters is automatically extracted via the dataset's labels.
    - measure and n_neighbors can be specified as (func, args_dict) tuples for custom strategies.
        - Do not specify context parameters in args_dict (data or adjacency matrix), they are provided by the pipeline.
    - To optimize a parameter via grid search, you can specify it as an iterable (e.g. list or np.arange).
    - The pipeline automatically searches through the product space of all parameters.
"""

default_params = {
    "n_neighbors": (log_neighbors, {"factor": 1}),
    "random_state": 42,  # Used for kmeans initialization. Has negligible effect for spectral methods.
    "affinity": "precomputed",  # Graph datasets provide adjacency directly.
    "n_it": 1,
    "assign_labels": "kmeans",
    "measure": (
        teleporting_undirected_measure,
        {"alpha": np.arange(0, 1.5, 0.1), "t": range(0, 25)},
    ),  # Grid search for GSC methods
    "tau": (
        avg_deg_taus,
        {"s": np.arange(-1, 1, 0.5)},
    ),  # Grid search for DI-SIM methods
    "metric_params": {
        "filter_coeffs": [
            {2: 0.5, 3: 0.5},
        ]
    },  # Grid search for graph-CH optimization
}

dataset_params = []


method_params = [
    ("SC-UN", {"laplacian_method": "unnorm", "standard": True, "measure": None}),
    ("SC-N", {"laplacian_method": "norm", "standard": True, "measure": None}),
    (
        "DSC+",
        {
            "gamma": np.arange(0, 1, 0.05),
        },
    ),
    (
        "DI-SIM-R",
        {
            "embedding": "right",
        },
    ),
    (
        "DI-SIM-L",
        {
            "embedding": "left",
        },
    ),
    (
        "DI-SIM-C",
        {
            "embedding": "combined",
        },
    ),
    ("GSC-N", {"laplacian_method": "norm"}),
    ("GSC-UN", {"laplacian_method": "unnorm"}),
    (
        "deg-GSC-N",
        {
            "laplacian_method": "norm",
            "measure": (degree_measure, {"gamma": np.arange(0, 1, 0.05)}),
        },
    ),
    (
        "deg-GSC-UN",
        {
            "laplacian_method": "unnorm",
            "measure": (degree_measure, {"gamma": np.arange(0, 1, 0.05)}),
        },
    ),
    ("uniform-GSC-N", {"laplacian_method": "norm", "measure": (uniform_measure, {})}),
    (
        "uniform-GSC-UN",
        {"laplacian_method": "unnorm", "measure": (uniform_measure, {})},
    ),
    (
        "perron-GSC-N",
        {"laplacian_method": "norm", "measure": (perron_vector_measure, {})},
    ),
    (
        "perron-GSC-UN",
        {"laplacian_method": "unnorm", "measure": (perron_vector_measure, {})},
    ),
]

method_dataset_params = []

"""
Do not edit below unless you really want to !
"""
config = ExperimentConfig(
    default_params=default_params,
    dataset_params=dataset_params,
    method_params=method_params,
    method_dataset_params=method_dataset_params,
)

if __name__ == "__main__":
    experiment(
        experiment_name=experiment_name,
        dataset_names=dataset_names,
        method_specs=method_specs,
        config=config,
        load_path=load_path,
        save_path=save_path,
        mode=mode,
        metrics=metrics,
        n_jobs=n_jobs,
        verbose=verbose,
    )
