import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dbcv
import time
import warnings
from itertools import product
from joblib import Parallel, delayed
from collections.abc import Iterable
from sklearn import cluster # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, calinski_harabasz_score # type: ignore
from scipy import stats
from utils.file_manager import load_dataset, save_experiment_results
from utils.logger import get_logger, set_logger_verbose
from competitors.disim import DiSim
from competitors.dsc import DSC

def experiment(experiment_name, dataset_names, method_specs, config, load_path, save_path, mode="score", metrics = ("nmi",), n_jobs=-1, verbose=True):
    """
    Run clustering experiments on multiple datasets and methods with parallel execution support.

    Parameters:
    -----------
    experiment_name : str
        Name for the experiment files and output directories
    dataset_names : list of str
        List of dataset names to process (must exist in load_path)
    method_specs : list of tuple
        List of (implicit_name, explicit_name) tuples for clustering methods.
        implicit_name: method identifier for clusterer() function
        explicit_name: human-readable method name for results
    config : ExperimentConfig
        Configuration object containing parameter hierarchies:
        default_params -> dataset_params -> method_params -> method_dataset_params
    load_path : str
        Path to the datasets folder (must exist)
    save_path : str
        Path where to save results (will be created if needed)
    mode : str, optional
        Experiment mode. Default: "score"
        - "score": Parallel scoring experiments 
        - "viz": Visualization experiments (2D datasets only)
        - "grid_search": Parameter optimization using GridSearchCV
    n_jobs : int, optional
        Number of parallel jobs for score mode. Default: -1 (all cores)
    verbose : bool, optional
        Control output verbosity. Default: True
        - True: Detailed progress, timing, scores, and error messages
        - False: Minimal output with just experiment start/completion status

    Returns:
    --------
    dict or pandas.DataFrame or None
        - Score mode: DataFrame (single metric) or dict of DataFrames (multiple metrics)
        - Viz mode: None (displays interactive plots)
        - Grid search mode: dict with grid search results for each method-dataset combination

    Raises:
    -------
    ValueError
        For invalid mode, unknown experiment type, or invalid metrics
    FileNotFoundError
        If load_path does not exist
    IOError
        If results cannot be saved

    Examples:
    ---------
    >>> # Simple NMI score experiment
    >>> results = experiment("test_exp", ["iris"], [("kmeans", "K-Means")],
    ...                      config, "datasets", "results", verbose=False)

    >>> # Multi-metric score experiment
    >>> results = experiment("comparison", dataset_list, method_list, config,
    ...                      "data", "output", mode=("score", ("nmi", "ari", "ch")))

    >>> # Visualization experiment
    >>> experiment("viz_exp", ["iris"], methods, config, "data", "plots",
    ...            mode="viz", verbose=True)

    >>> # Grid search experiment
    >>> results = experiment("grid_exp", dataset_list, method_list, config,
    ...                      "data", "output", mode="grid_search", 
    ...                      metrics=("nmi", "ari"))
    """

    if mode not in ("score", "viz", "grid_search"):
        raise ValueError(f"Unknown experiment mode: {mode}. Must be 'score', 'viz', or 'grid_search'")

    if not isinstance(metrics, (str,tuple,list)) :
        raise ValueError("Metrics must be a string (for a single metric) or a tuple/list of strings.")

    valid_metrics = {"nmi", "ari", "ami", "ch", "dbcv"}
    invalid_metrics = set(metrics) - valid_metrics
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}. Valid metrics are: {valid_metrics}")

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Dataset path '{load_path}' does not exist")

    os.makedirs(save_path, exist_ok=True)

    set_logger_verbose(verbose)
    logger = get_logger()

    logger.minimal(f"Running experiment '{experiment_name}'...")
    
    if verbose:
        logger.section("Experiment Configuration")
        logger.list_items("Datasets", dataset_names)
        logger.list_items("Methods", [spec[1] for spec in method_specs])
        logger.list_items("Metrics", list(metrics))
        if mode == "score":
            logger.info(f"Parallel execution with {n_jobs} jobs")

    if mode == "score":
        results_dict, all_parameters = _run_score_experiment(dataset_names, method_specs, config, load_path, metrics, n_jobs)
        fig, grid_results = None, None

    elif mode == "viz":
        fig, all_parameters = _run_viz_experiment(dataset_names, method_specs, config, load_path, metrics)
        results_dict, grid_results = None, None

    else :# mode == "grid_search"
        grid_results, all_parameters = _run_grid_search_experiment(dataset_names, method_specs, config, load_path, metrics, n_jobs)
        fig, results_dict = None, None
    
    try:
        save_experiment_results(mode=mode,experiment_name=experiment_name, save_path=save_path, parameters=all_parameters, metric_dfs=results_dict,figure=fig, grid_results=grid_results)        
        logger.success(f"{mode.capitalize().replace('_',' ')} experiment '{experiment_name}' completed and saved successfully to '{save_path}'.")
    except Exception as e:
        raise IOError(f"Failed to save experiment results: {str(e)}")

def clusterer(method_name, params):
    """Create sklearn-compatible clustering objects for specified methods.

    Each clustering object must have a fit() or fit_predict() method and either
    a labels_ attribute or predict() method for retrieving cluster assignments.

    Parameters:
    -----------
    method_name : str
        Name of the clustering method. Supported values:
        - "spectral": SpectralClustering with custom parameters
        - "kmeans": KMeans clustering
    params : dict
        Dictionary of parameters to pass to the clustering algorithm.
        See individual method documentation for supported parameters.

    Returns:
    --------
    clustering_object
        Sklearn-compatible clustering object with fit() method and labels_ attribute

    Raises:
    -------
    ValueError
        If method_name is not supported

    Notes:
    ------
    To add new clustering methods:
    1. Add the method implementation in competitors/competitors.py
    2. Ensure it follows sklearn API (fit method, labels_ attribute)
    3. Add it as a new elif branch in this function
    4. Update the supported methods list in this docstring

    Supported Parameters:
    ---------------------
    For "spectral":
        - n_clusters: Number of clusters (default: 3)
        - n_neighbors: Number of neighbors for nearest neighbors graph (default: 6)
        - affinity: Affinity method (default: "nearest_neighbors")
        - laplacian_method: Laplacian normalization method (default: "norm")
        - measure: Custom vertex measure function (default: None)
        - standard: Whether to use standard spectral clustering (default: False)
        - random_state: Random state for reproducibility (default: 42)

    For "kmeans":
        - n_clusters: Number of clusters (default: 3)
        - random_state: Random state for reproducibility (default: 42)
    """
    if method_name == "spectral":
        return cluster.SpectralClustering(
            n_clusters=params.get("n_clusters", 3),
            n_neighbors=params.get("n_neighbors", 6),
            affinity=params.get("affinity", "nearest_neighbors"),
            laplacian_method=params.get("laplacian_method", "norm"),
            measure=params.get("measure", None),
            standard=params.get("standard", False),
            assign_labels=params.get("assign_labels", "kmeans"),
            random_state=params.get("random_state", 42),
            eigen_solver=params.get("eigen_solver", "arpack"),
            #eigen_tol=params.get("eigen_tol", None)
            )

    elif method_name == "di_sim":
        return DiSim(
            n_clusters=params.get("n_clusters", 3),
            n_neighbors=params.get("n_neighbors", 6),
            tau=params.get("tau", 1e-8),
            embedding=params.get("embedding", "left"),
            epsilon=params.get("epsilon", 1e-8),
            random_state=params.get("random_state", 42)
            )
    
    elif method_name == "dsc":
        return DSC(
            n_clusters=params.get("n_clusters", 3),
            n_neighbors=params.get("n_neighbors", 6),
            gamma=params.get("gamma", 0.5),
            max_iter=params.get("max_iter", 100),
            tol=params.get("tol", 1e-4),
            epsilon=params.get("epsilon", 1e-8),
            random_state=params.get("random_state", 42)
        )
    
    elif method_name == "kmeans":
        return cluster.KMeans(
            n_clusters=params.get("n_clusters", 3),
            random_state=params.get("random_state", 42)
            )
    
    elif method_name == "hdbscan":
        return cluster.HDBSCAN(
            min_cluster_size=params.get("min_cluster_size", 5),
            min_samples=params.get("min_samples", None),
            metric=params.get("metric", "euclidean"),
            cluster_selection_method=params.get("cluster_selection_method", "eom"),
            allow_single_cluster=params.get("allow_single_cluster", False),
        )
    else:
        raise ValueError(f"Unknown clustering method: {method_name}. Supported methods: 'spectral', 'kmeans', 'hdbscan")
    
def _run_grid_search_experiment(dataset_names, method_specs, config, load_path, metrics, n_jobs):
    """Run grid search experiments on multiple datasets and methods with parallel execution."""
    
    logger = get_logger()
    parallel_verbose = 2 if logger.verbose else 0

    grid_results = {}
    all_parameters = {dataset_name: {} for dataset_name in dataset_names}
    
    for i, dataset_name in enumerate(dataset_names):
        if i > 0:
            logger.info("─" * 50)
            
        try:
            X, y = load_dataset(load_path, dataset_name)
            logger.info(f"Loaded {dataset_name} dataset: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            continue
            
        dataset_results = {}
        
        for implicit_name, explicit_name in method_specs:
            logger.info(f"Running grid search for: {explicit_name}")
                
            resolved_params = config.get_final_params(y, dataset_name, explicit_name, implicit_name)
            all_parameters[dataset_name][explicit_name] = resolved_params.copy()
            
            grid_params, base_params, param_names, combinations = _resolve_grid_params(resolved_params)
            
            if not grid_params:
                logger.info(f"  No grid parameters specified for {explicit_name}, running single combination...")
                combinations = [()] 
                param_names = []
            else:
                param_str = ", ".join(param_names)
                logger.info(f"  Grid parameters: {param_str}")
            
            logger.info(f"  Total combinations: {len(combinations)}")
            
            try:
                n_it = resolved_params.get("n_it", 1)
                assert isinstance(n_it, int) and n_it > 0
                logger.info(f"  Iterations per combination: {n_it}")
            except Exception as e:
                logger.error(f"Invalid n_it parameter for {explicit_name} on {dataset_name}: {n_it}. Must be a positive integer.")
                continue

            tasks = []
            for combination in combinations:
                current_params = base_params.copy()
                current_params.update(dict(zip(param_names, combination)))
                
                
                task_config = copy.deepcopy(config)
                if explicit_name not in task_config.method_dataset_params_dict:
                    task_config.method_dataset_params_dict[explicit_name] = {}
                
                #Update the internal parameters at the highest precedence, so task_config.get_final_params() correctly extracts those params.
                task_config.method_dataset_params_dict[explicit_name][dataset_name] = current_params
                
                task = {
                    'dataset_name': dataset_name,
                    'method_spec': (implicit_name, explicit_name),
                    'config': task_config,
                    'X': X,
                    'y': y,
                    'metrics': metrics,
                }
                tasks.append(task)
            
            start_time = time.time()
            results = Parallel(n_jobs=n_jobs, verbose=parallel_verbose)(
                delayed(_run_single_task)(task, mode="grid_search") for task in tasks
            )
            timing = time.time() - start_time
            method_results = _aggregate_grid_search_results(results, metrics)
            method_results['timing'] = timing
            method_results['grid_param_names'] = param_names  
            method_results['all_grid_results'] = results  
            
            dataset_results[explicit_name] = method_results
            
            logger.info(f"  Completed {method_results['successful_combinations']} combinations in {timing:.2f}s")
            
            if len(metrics) == 1:
                target_metric = list(metrics)[0]
                if target_metric in method_results:
                    best_score_info = method_results[target_metric]['best']
                    logger.info(f"  → Best {target_metric.upper()}: {best_score_info['mean']:.3f} ± {best_score_info['std']:.3f}")
            else:
                logger.info("  → Best scores:")
                for target_metric in metrics:
                    if target_metric in method_results:
                        best_score_info = method_results[target_metric]['best']
                        logger.info(f"    • {target_metric.upper()}: {best_score_info['mean']:.3f} ± {best_score_info['std']:.3f}")
        
        grid_results[dataset_name] = dataset_results
    
    return grid_results, all_parameters

def _run_score_experiment(dataset_names, method_specs, config, load_path, metrics, n_jobs):
    """Run clustering score experiments in parallel across datasets and methods."""

    logger = get_logger()
    
    tasks=[]
    for i, dataset_name in enumerate(dataset_names):
        if i > 0 :
            logger.info("─" * 50)
            
        logger.info(f"Loading {dataset_name} dataset...")
        try: 
            X, y = load_dataset(load_path, dataset_name)
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            continue

        for method_spec in method_specs:
            task = {
                    'dataset_name': dataset_name,
                    'method_spec': method_spec ,
                    'config': config,
                    'X': X,
                    'y': y,
                    'metrics': metrics, 
                    }
            tasks.append(task)

    parallel_verbose = 2 if logger.verbose  else 1

    results = Parallel(n_jobs=n_jobs, verbose=parallel_verbose)(
        delayed(_run_single_task)(task, mode="score") for task in tasks
    )
    explicit_method_names = [method[1] for method in method_specs]

    dfs = {}
    for metric in metrics:
        all_cols = []
        for name in explicit_method_names:
            all_cols.extend([f"{name}_mean", f"{name}_std"])
        
        dfs[metric] = pd.DataFrame(index=dataset_names, columns=all_cols, dtype=float)

    all_parameters = {dataset_name: {} for dataset_name in dataset_names}

    for result in results :
        if result is not None:
            dataset_name = result["dataset_name"]
            explicit_name = result["explicit_name"]
            scores = result["scores"]
            final_params = result["final_params"]

            for metric in metrics:
                score_data = scores.get(metric, {'mean': 0.0, 'std': 0.0})
                mean_val = score_data.get('mean', 0.0)
                std_val = score_data.get('std', 0.0)
             
                
                dfs[metric].loc[dataset_name, f"{explicit_name}_mean"] = mean_val
                dfs[metric].loc[dataset_name, f"{explicit_name}_std"] = std_val

            all_parameters[dataset_name][explicit_name] = final_params

    results_dict = {}
    for metric in metrics:
        results_dict[metric] = dfs[metric]

    return results_dict, all_parameters

def _run_viz_experiment(dataset_names, method_specs, config, load_path, metrics):
    """Run clustering experiments with visualization."""
    logger = get_logger()
    num_datasets = len(dataset_names)
    num_methods = len(method_specs)
    all_parameters = {}

    plt.figure(figsize=(num_methods * 3 + 1, num_datasets * 3 + 1))
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01)

    plot_num = 1

    for i_dataset, dataset_name in enumerate(dataset_names):
        if i_dataset > 0: 
            logger.info("─" * 50)
            
        try:
            X, y = load_dataset(load_path, dataset_name)
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            plot_num += len(method_specs)
            continue

        if X.shape[1] != 2:
            logger.info(f"Warning: Dataset {dataset_name} is not 2D (shape: {X.shape}). Skipping visualization.")
            plot_num += len(method_specs)
            continue

        X = StandardScaler().fit_transform(X)
        all_parameters[dataset_name] = {}

        logger.info(f"Processing dataset: {dataset_name}")

        for j_method, method_spec in enumerate(method_specs):
            implicit_name, explicit_name = method_spec

            logger.info(f"Running {explicit_name}...")

            plt.subplot(num_datasets, num_methods, plot_num)
            params = config.get_final_params(y, dataset_name, explicit_name, implicit_name)
            all_parameters[dataset_name][explicit_name] = params

            t0 = time.time()
            y_pred, scores, error = _process_method_on_dataset(
                dataset_name, method_spec, params, X=X, y=y, metrics=metrics
            )
            t1 = time.time()

            if error:
                logger.error(f"Error processing {explicit_name} on {dataset_name}: {error}")
                plt.text(0.5, 0.5, "Error", transform=plt.gca().transAxes,
                        horizontalalignment="center", verticalalignment="center")
                plt.xticks(())
                plt.yticks(())
                plot_num += 1
                continue
            
            _create_cluster_plot(X, y_pred, explicit_name, dataset_name,
                                t1-t0, i_dataset, j_method,  scores, metrics)

            score_strs = []
            for metric in metrics:
                score_value = scores.get(metric, 0.0)
                score_strs.append(f"{metric.upper()}={score_value:.1f}")
            logger.info(f"{explicit_name}: {', '.join(score_strs)} in {t1-t0:.2f}s")

            plot_num += 1

    fig = plt.gcf()
    plt.show()

    return fig, all_parameters

def _run_single_task(task, mode):
    """Execute a single dataset-method combination with support for both score and grid search modes, for a specified number of iterations."""
    
    logger = get_logger()
    dataset_name = task['dataset_name']
    method_spec = task['method_spec']
    config = task['config']
    X = task['X']
    y = task['y']
    metrics = task['metrics']
    implicit_name, explicit_name = method_spec
    params = config.get_final_params(y, dataset_name, explicit_name, implicit_name)

    try:
        n_it = params.get("n_it",1)
        assert isinstance(n_it, int) and n_it > 0
    except Exception as e:
        logger.error(f"Invalid n_it parameter for {explicit_name} on {dataset_name}: {n_it}. Must be a positive integer.")
        return None
    
    all_scores = []
    all_labels = []
     
    for i in range(n_it):

        y_pred, iteration_scores, iteration_error = _process_method_on_dataset(
            dataset_name, method_spec, params, X, y, metrics=metrics)
        
        if iteration_error:
            logger.error(f"Error processing {explicit_name} on {dataset_name}: {iteration_error}")
            return None
        
        all_scores.append(iteration_scores)
        all_labels.append(y_pred.tolist())  # Convert to list for JSON serialization
    
        
    if all_scores:
        scores = dict()
        for metric in metrics:
            metric_values = [score_dict[metric] for score_dict in all_scores]
            scores[metric] = {
                'mean': round(np.mean(metric_values),4),
                'std': round(np.std(metric_values),4)
            }
    else:
        scores = {metric: {'mean': 0.0, 'std': 0.0} for metric in metrics}

    if mode=="score" and logger.verbose:
        score_strs = []
        for metric in metrics:
                mean_value = scores[metric]['mean']
                std_value = scores[metric]['std']
                score_strs.append(f"{metric.upper()}={mean_value:.1f}±{std_value:.1f}")

        logger.info(f"{explicit_name} on {dataset_name}: {', '.join(score_strs)}")

    result = {"scores": scores, "final_params": params, "error": None, "predicted_labels": all_labels}

    if mode=="score":
        result["dataset_name"]=dataset_name
        result["explicit_name"]=explicit_name
    
    return result

def _process_method_on_dataset(dataset_name, method_spec, params, X, y, metrics):
    """Process a single method on a dataset, returning clustering results."""
    implicit_name, explicit_name = method_spec

    try:
        cluster_obj = clusterer(implicit_name, params)

        with warnings.catch_warnings():
            _ignore_warnings()
            if hasattr(cluster_obj, 'fit_predict'):
                y_pred = cluster_obj.fit_predict(X)
            elif hasattr(cluster_obj, 'fit'):
                cluster_obj.fit(X)
                y_pred = cluster_obj.labels_ if hasattr(cluster_obj, "labels_") else cluster_obj.predict(X)
            else:
                raise ValueError(f"Clustering object for {implicit_name} has neither fit nor fit_predict method")

        y_pred = y_pred.astype(int)

        scores = _compute_clustering_scores(y, y_pred, metrics, X)

        return y_pred, scores, None

    except Exception as e:
        error = str(e)
        return None, None, error
    
def _compute_clustering_scores(y_true, y_pred, metrics, X):
    """Compute clustering scores for specified metrics.
    Available metrics:

    - Supervised :
        - "nmi": Normalized Mutual Information
        - "ari": Adjusted Rand Index
        - "ami": Adjusted Mutual Information
    - Unsupervised :
        - "ch": Calinski-Harabasz index (requires data matrix X)
    """
    scores = {}

    if y_true is None or len(np.unique(y_true)) <= 1:
        for metric in metrics:
            scores[metric] = 0.0
        return scores

    n_clusters = len(np.unique(y_pred))
    if n_clusters <= 1:
        for metric in metrics:
            scores[metric] = 0.0
        return scores


    for metric in metrics:
        if metric == "nmi":
            score = normalized_mutual_info_score(y_true, y_pred)
        elif metric == "ari":
            score = adjusted_rand_score(y_true, y_pred)
        elif metric == "ami":
            score = adjusted_mutual_info_score(y_true, y_pred)
        elif metric == "ch":
            if X is None:
                score = 0.0  
            else:
                score = calinski_harabasz_score(X, y_pred)
        elif metric == "dbcv":
            score = dbcv.dbcv(X, y_pred, check_duplicates=False)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores[metric] = round(score , 4)

    return scores

def _create_cluster_plot(X, y_pred, explicit_name, dataset_name, timing,
                        i_dataset, j_method, scores, metrics):
    """Create a single cluster plot in the visualization grid."""
    colors = np.array([
        "#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
        "#984ea3", "#999999", "#e41a1c", "#dede00"
    ])

    if i_dataset == 0:
        plt.title(explicit_name, size=14)

    point_colors = colors[y_pred % len(colors)]
    plt.scatter(X[:, 0], X[:, 1], s=10, color=point_colors)

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())

    if j_method == 0:
        plt.ylabel(dataset_name, rotation=90, size=12)

    plt.text(0.99, 0.01, f"{timing:.2f}s".lstrip("0"),
            transform=plt.gca().transAxes, size=12,
            horizontalalignment="right", verticalalignment="bottom")


    score_lines = []
    for metric in metrics:
        score_value = scores.get(metric, 0.0)
        score_lines.append(f"{metric.upper()}: {score_value:.1f}")

    score_text = "\n".join(score_lines)
    plt.text(0.01, 0.99, score_text,
            transform=plt.gca().transAxes, size=10,
            horizontalalignment="left", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def _resolve_grid_params(resolved_params):
    """Resolve grid search parameters by expanding callables and creating parameter combinations."""
    
    # This is not very clean nor readable, but works. Difficulty is looking for the grid params in the args of callable params, then the normal grid params, without repeating code.
    # Will try to refactor.
    expanded_params = {}
    
    # First pass: expand callable parameters with grid search kwargs
    for param_name, param_value in resolved_params.items():
        if isinstance(param_value, tuple) and len(param_value) == 2 and callable(param_value[0]) and isinstance(param_value[1], dict):
            func, kwargs = param_value
            
            # Separate grid kwargs from fixed kwargs
            grid_kwargs = {}
            fixed_kwargs = {}
            
            for arg, arg_value in kwargs.items():
                if isinstance(arg_value, Iterable) and not isinstance(arg_value, str) and len(arg_value) > 1:
                    grid_kwargs[arg] = arg_value
                else:
                    fixed_kwargs[arg] = arg_value
            
            # If there are grid parameters in kwargs, create combinations
            if grid_kwargs:
                grid_keys = list(grid_kwargs.keys())
                grid_values = list(grid_kwargs.values())
                combinations = list(product(*grid_values))
                
                # Create list of (func, kwargs_dict) tuples for each combination
                func_combinations = []
                for combination in combinations:
                    current_kwargs = fixed_kwargs.copy()
                    current_kwargs.update(dict(zip(grid_keys, combination)))
                    func_combinations.append((func, current_kwargs))
                
                expanded_params[param_name] = func_combinations
                continue
        
        expanded_params[param_name] = param_value
    
    #Second pass : treat normal params (not callables)
    grid_params = {}
    base_params = {}
    
    for param_name, param_value in expanded_params.items():
        if not(isinstance(param_value, tuple) and len(param_value) == 2 and callable(param_value[0]) and isinstance(param_value[1], dict)): #If not (callable, dict) 
            # Previous condition needed because (callable, dict) fverifies this one while not being a grid param
            if isinstance(param_value, Iterable) and not isinstance(param_value, str) and len(param_value) > 1: 
                grid_params[param_name] = param_value
        else:
                base_params[param_name] = param_value
        
    if grid_params:
        param_names = list(grid_params.keys())
        param_values = list(grid_params.values())
        combinations = list(product(*param_values))
    else:
        param_names = []
        combinations = []
    
    return grid_params, base_params, param_names, combinations

def _aggregate_grid_search_results(results, metrics):
    """Aggregate grid search results to compute mean, std, best, and best_params for each metric."""
    successful_results = [r for r in results if r is not None]
    error_count = len(results) - len(successful_results)
    
    if not successful_results:
        return {'error': f"All {len(results)} parameter combinations failed"}
    
    aggregated = {
        'successful_combinations': len(successful_results),
        'failed_combinations': error_count,
        'total_combinations': len(results)
    }
    
    for metric in metrics:
        metric_scores = []
        metric_params = []
        metric_all_results = [] 
        
        for result in successful_results:
            if metric in result['scores']:
                score_data = result['scores'][metric]
                if isinstance(score_data, dict) and 'mean' in score_data:
                    metric_scores.append(score_data['mean'])
                else:
                    metric_scores.append(score_data)
                    
                metric_all_results.append({
                    'params': result['final_params'],
                    'scores': result['scores'],
                    'predicted_labels': result.get('predicted_labels', [])  # Include labels
                })
                metric_params.append(result['final_params'])
        
        if metric_scores:
            best_idx = np.argmax(metric_scores)
            best_params = metric_params[best_idx]
            best_full_scores = successful_results[best_idx]['scores']
            best_score_dict = best_full_scores[metric]
            best_labels = successful_results[best_idx].get('predicted_labels', [])
            
            aggregated[metric] = {
                'mean_across_grid': round(np.mean(metric_scores), 4),
                'std_across_grid': round(np.std(metric_scores), 4),
                'best': best_score_dict, 
                'best_params': best_params,
                'best_scores': best_full_scores,
                'best_predicted_labels': best_labels,  # Include best labels
                'n_scores': len(metric_scores),
                'all_results': metric_all_results  
            }
    
    return aggregated

def _ignore_warnings():
    warnings.filterwarnings("ignore",
        message="the number of connected components of the connectivity matrix is [0-9]{1,2} > 1. Completing it to avoid stopping the tree early.",
        category=UserWarning)
    warnings.filterwarnings("ignore",
        message="Graph is not fully connected, spectral embedding may not work as expected.",
        category=UserWarning)



