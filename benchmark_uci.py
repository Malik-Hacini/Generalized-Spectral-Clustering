"""
Clustering benchmark performed in the GSC paper.
"""
import numpy as np
import time
from utils.config import ExperimentConfig
from utils.experiments_utils import experiment
from competitors.neighbors import log_neighbors
from competitors.measures import teleporting_undirected_measure
from competitors.disim import avg_deg_taus

"""
Basic experiment config:
"""
save_path = "results"
experiment_name = "benchmark_uci"
mode = "grid_search" # Either "score", "grid_search" or "viz" when all datasets are 2D.
metrics = ("ami", "ch") # Valid metrics: "ami", "ari", "nmi", "ch"
n_jobs = -1  # Number of parallel jobs (set to 1 for single-threaded execution, -1 to use all available cores)
verbose = True

"""
Datasets and methods configuration:
"""
load_path = "datasets"
dataset_names = ["breast_tissue", "wine", "control_chart", "glass", "iris", "parkinsons", "seeds", "segmentation", "vertebral", "wdbc", "yeast"]
method_specs = [
    ("spectral", "SC-UN"),
    ("spectral", "SC-N"),
    ("dsc", "DSC+"),
    ("di_sim", "DI-SIM-R"),
    ("di_sim", "DI-SIM-L"),
    ("di_sim", "DI-SIM-C"),
    ("spectral", "GSC-N"),
    ("spectral", "GSC-UN"), 
 # (Internal  name, Display name ) - see utils.experiments_utils.clusterer
]

#Parameters
"""
Parameter hierarchy (lowest to highest precedence):
1. default_params : Base parameters for all experiments
2. General dataset parameters : [(dataset_name, params_dict), ...]
3. General method parameters: [(method_name, params_dict), ...]
4. Specific parameters for a method/dataset combination : [(method_name, [(dataset, params), ...]), ...]

Notes :
    - The number of clusters is automatically extracted via the dataset's labels.
    - measure and n_neighbors can be specified as (func, args_dict) tuples for custom strategies. 
        - Do not pecify context parameters in args_dict (data or adjacency matrix), they are provided by the pipeline.
    - To optimize a parameter via grid search, you can specify it as an iterable (e.g. list or np.arange). 
    - The pipeline automatically searches through the product space of all parameters.
"""

default_params = {
    "n_neighbors": (log_neighbors, {"factor": 1}),
    "random_state": 42, #Used for kmeans initialization. Has negligible effect for spectral methods.
    "affinity": "nearest_neighbors",
    "n_it": 1,  
    "assign_labels": "kmeans",
    "measure": (teleporting_undirected_measure, {'alpha': np.arange(0, 1.5, 0.1), 't': range(0,25)}), # Grid search for GSC methods
    "tau": (avg_deg_taus, {"s": np.arange(-1, 1, 0.5)}), # Grid search for DI-SIM methods
}

dataset_params = []


method_params = [
    ("SC-UN", { 
        "laplacian_method": "unnorm", 
        "standard": True,
        "measure": None
    }),

    ("SC-N", {
        "laplacian_method": "norm", 
        "standard": True,
        "measure": None
    }),

    ("DSC+", {
        "gamma": np.arange(0, 1, 0.05),
    }),

    ("DI-SIM-R", {
        "embedding": "right",
    }),

    ("DI-SIM-L", {
        "embedding": "left",
    }),

    ("DI-SIM-C", {
        "embedding": "combined",
    }),

    ("GSC-N",{
        "laplacian_method": "norm"
    }),

    ("GSC-UN",{
        "laplacian_method": "unnorm"
    })
]

method_dataset_params = [
 ]

"""
Do not edit below unless you really want to !
"""
config = ExperimentConfig(
    default_params=default_params,
    dataset_params=dataset_params,
    method_params=method_params,
    method_dataset_params=method_dataset_params
)

if __name__ == "__main__":

    start = time.time()
    results_df_parallel = experiment(
        experiment_name=experiment_name,
        dataset_names=dataset_names,
        method_specs=method_specs,
        config=config,
        load_path=load_path,
        save_path=save_path,
        mode=mode,
        metrics=metrics,
        n_jobs=n_jobs,
        verbose=verbose
    )
    end = time.time()
    print(f"Experiment completed in {end - start} seconds.")
