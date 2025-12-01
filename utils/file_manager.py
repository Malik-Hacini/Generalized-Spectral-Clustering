import os
import json
import numpy as np
import pandas as pd
from datasets import load_from_disk, Dataset, DatasetDict
from utils.logger import get_logger


def load_dataset(path: str, name: str, split: str = "train", feature_cols=None, label_col: str = "labels"):
    """
    Load a clustering dataset from local disk storage and return feature and label arrays.

    This function loads datasets stored in Hugging Face format for use in clustering
    experiments.
    Parameters:
    -----------
    path : str
        Base directory path containing datasets (e.g., "datasets")
    name : str
        Name of the specific dataset folder (e.g., "iris", "wine")
    split : str, optional
        Dataset split to load. Default: "train"
        Falls back to "default" if split not found
    feature_cols : list of str, optional
        Names of feature columns to load. If None, loads all columns except label_col
    label_col : str, optional
        Name of the column containing true cluster labels. Default: "labels"

    Returns:
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : numpy.ndarray
        True cluster labels of shape (n_samples,)

    Raises:
    -------
    ValueError
        If dataset directory structure is invalid or split not found
    FileNotFoundError
        If dataset path does not exist

    Examples:
    ---------
    >>> # Load iris dataset
    >>> X, y = load_dataset("datasets", "iris")
    >>> print(X.shape, y.shape)
    (150, 4) (150,)

    >>> # Load specific features only
    >>> X, y = load_dataset("datasets", "wine", feature_cols=["feature1", "feature2"])

    --------
    Expected file structure:
    ```
    path/
    └── name/
        ├── dataset_info.json
        ├── state.json
        └── train/  (or split/ or default/)
            ├── data-00000-of-00001.arrow
            └── dataset_info.json
    ```
    The dataset must be stored in Hugging Face Dataset format with:
    - dataset_info.json file
    - Arrow format data files

    ------
    """

    target_path = os.path.join(path, name)
    if not os.path.isfile(os.path.join(target_path, "dataset_info.json")):
        candidate = os.path.join(target_path, split)
        if os.path.isdir(candidate):
            target_path = candidate
        else:
            candidate = os.path.join(target_path, "default")
            if os.path.isdir(candidate):
                target_path = candidate
            else:
                raise ValueError(f"No dataset found in '{target_path}'. "
                                 f"Expected '{target_path}/{split}' or '{target_path}/default' to exist.")

    ds = load_from_disk(target_path)

    if hasattr(ds, "get"):
        if split not in ds:
            raise ValueError(f"Split '{split}' not found in dataset at {path}. Available splits: {list(ds.keys())}")
        ds = ds[split]

    if feature_cols is None:
        feature_cols = [c for c in ds.column_names if c != label_col]

    ds_numpy = ds.with_format("numpy", columns=feature_cols + [label_col])
    batch = ds_numpy[:]

    X = np.column_stack([batch[c] for c in feature_cols])
    y = batch[label_col]

    return X, y

def save_dataset(data: np.ndarray, labels: np.ndarray,
                 path: str, name: str,
                 feature_cols: list[str] = None,
                 label_col: str = "labels") -> None:
    """
    Save dataset in Hugging Face format for use in experiments.

    Parameters:
    -----------
    data : numpy.ndarray
        Feature matrix of shape (n_samples, n_features) or (n_samples,) for 1D data
    labels : numpy.ndarray
        Cluster labels of shape (n_samples,)
    path : str
        Base directory where to save the dataset (e.g., "datasets")
    name : str
        Dataset name, will create subdirectory path/name/
    feature_cols : list of str, optional
        Names for feature columns. If None, auto-generates ['f0', 'f1', ...]
        Length must match data.shape[1]
    label_col : str, optional
        Column name for labels. Default: "labels"

    Raises:
    -------
    AssertionError
        If data and labels have different number of samples
    OSError
        If directory creation fails

    ------

    Examples:
    ---------
    >>> # Save a simple 2D dataset
    >>> import numpy as np
    >>> X = np.random.rand(100, 2)
    >>> y = np.random.randint(0, 3, 100)
    >>> save_dataset(X, y, "datasets", "my_dataset")

    >>> # Save with custom column names
    >>> save_dataset(X, y, "datasets", "custom",
    ...              feature_cols=["x_coord", "y_coord"],
    ...              label_col="cluster_id")

    --------
    File Structure:
    ```
    path/
    └── name/
        ├── dataset_info.json
        ├── state.json
        └── train/
            ├── data-00000-of-00001.arrow
            └── dataset_info.json
    ```

    The saved dataset can be loaded using load_dataset(path, name).
    """
    data, labels = np.asarray(data), np.asarray(labels)
    assert data.shape[0] == labels.shape[0], "data and labels must have same number of samples"

    if feature_cols is None:
        if data.ndim == 1:
            feature_cols = ["feature"]
            data = data.reshape(-1, 1)
        else:
            feature_cols = [f"f{i}" for i in range(data.shape[1])]

    data_dict = {feature_cols[i]: data[:, i] for i in range(data.shape[1])}
    data_dict[label_col] = labels

    ds = Dataset.from_dict(data_dict)

    ds_dict = DatasetDict({"train": ds})

    out_dir = os.path.join(path, name, "train")
    os.makedirs(out_dir, exist_ok=True)

    ds_dict.save_to_disk(os.path.join(path, name))

def save_experiment_results(mode: str, experiment_name: str, save_path: str, 
                           parameters: dict = None,
                           metric_dfs: dict = None, figure = None, 
                           grid_results: dict = None) -> None:
    """
    Save experiment results in organized directory structure for all modes.

    Parameters:
    -----------
    mode : str
        Experiment mode: "score", "viz", or "grid_search"
    experiment_name : str
        Name of the experiment (will be used as folder name)
    save_path : str
        Base directory path where to save the files (e.g., 'results')
    parameters : dict, optional
        Nested dictionary with parameters used for each dataset and method.
        Structure: {dataset_name: {method_name: params_dict}}
        Required for all modes to save the parameter configuration
    metric_dfs : dict, optional
        Dictionary mapping metric names to DataFrames (required for "score" mode)
    figure : matplotlib.figure.Figure, optional
        Figure to save (required for "viz" mode)
    grid_results : dict, optional
        Grid search results dictionary (required for "grid_search" mode)

    Directory Structure:
    -------------------
    For mode="score":
    ```
    save_path/
    └── {experiment_name}_scores/
        ├── {experiment_name}_scores_nmi.csv
        ├── {experiment_name}_scores_ari.csv
        ├── {experiment_name}_scores_ami.csv
        ├── {experiment_name}_scores_ch.csv
        └── {experiment_name}_params.json
    ```

    For mode="viz":
    ```
    save_path/
    └── {experiment_name}_viz/
        ├── {experiment_name}_visualization.png
        └── {experiment_name}_params.json
    ```

    For mode="grid_search":
    ```
    save_path/
    └── {experiment_name}_grid_search/
        ├── {experiment_name}_params.json
        ├── dataset1/
        │   ├── dataset1_summary.csv
        │   ├── method1/
        │   │   ├── method1_all_results.json
        │   │   └── method1_best_results.json
        │   └── method2/
        │       ├── method2_all_results.json
        │       └── method2_best_results.json
        └── dataset2/
            ├── dataset2_summary.csv
            ├── method1/
            │   ├── method1_all_results.json
            │   └── method1_best_results.json
            └── method2/
                ├── method2_all_results.json
                └── method2_best_results.json
    ```
    
    """
    
    logger = get_logger()
    dir_name = f"{experiment_name}_{mode}"
    
    experiment_dir = os.path.join(save_path, dir_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    if logger.verbose:
        logger.info(f"Saving {mode} experiment results to: {experiment_dir}")
    
    if mode == "score":
        _save_score_files(experiment_dir, experiment_name, metric_dfs)
    elif mode == "viz":
        _save_viz_files(experiment_dir, experiment_name, figure)
    elif mode == "grid_search":
        _save_grid_search_files(experiment_dir, grid_results)
    
    if parameters:
        params_filename = f"{experiment_name}_params.json"
        params_path = os.path.join(experiment_dir, params_filename)
        with open(params_path, 'w') as f:
            json.dump(parameters, f, indent=2, default=str)
        
        logger.info(f"  Parameters saved: {params_filename}")
    
def _save_score_files(experiment_dir: str, experiment_name: str, metric_dfs: dict):
    """Save score mode specific files."""
    
    logger = get_logger()
    for metric, df in metric_dfs.items():
        csv_filename = f"{experiment_name}_scores_{metric}.csv"
        csv_path = os.path.join(experiment_dir, csv_filename)
        df.to_csv(csv_path, index=True)
        
        logger.info(f"  Scores saved: {csv_filename}")

def _save_viz_files(experiment_dir: str, experiment_name: str, figure):
    """Save visualization mode specific files."""
    
    logger = get_logger()
    plot_filename = f"{experiment_name}_visualization.png"
    plot_path = os.path.join(experiment_dir, plot_filename)
    figure.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"  Visualization saved: {plot_filename}")

def _save_grid_search_files(experiment_dir: str, grid_results: dict):
    """Save grid search mode specific files."""
    
    logger = get_logger()
    for dataset_name, dataset_results in grid_results.items():
        logger.info(f"  Saving results for dataset: {dataset_name}")
            
        dataset_dir = os.path.join(experiment_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        summary_rows = []
        
        for method_name, method_results in dataset_results.items():
            if 'error' in method_results:
                continue
                
            summary_row = {'method': method_name}
            
            for key, value in method_results.items():
                if isinstance(value, dict) and 'best' in value:
                    metric_name = key
                    summary_row[f"{metric_name}_mean_across_grid"] = round(value.get('mean_across_grid', 0), 3)
                    summary_row[f"{metric_name}_std_across_grid"] = round(value.get('std_across_grid', 0), 3)
                    best_score = value['best']
                    summary_row[f"{metric_name}_best_mean"] = round(best_score['mean'], 3)
                    summary_row[f"{metric_name}_best_std"] = round(best_score['std'], 3)
                   
                    
            summary_rows.append(summary_row)
            
            grid_param_names = method_results.get('grid_param_names', [])
            all_grid_results = method_results.get('all_grid_results', [])
            
            all_results = []
            for result in all_grid_results:
                if result.get('error') is None and 'final_params' in result and 'scores' in result:
                    grid_params = {param: result['final_params'][param] 
                                 for param in grid_param_names 
                                 if param in result['final_params']}
                    
                    detail_entry = grid_params.copy()
                    detail_entry.update(result['scores'])
                    # Include predicted labels if available
                    if 'predicted_labels' in result:
                        detail_entry['predicted_labels'] = result['predicted_labels']
                    all_results.append(detail_entry)
            
            method_dir = os.path.join(dataset_dir, method_name)
            os.makedirs(method_dir, exist_ok=True)
            
            if all_results:
                all_results_file = os.path.join(method_dir, f"{method_name}_all_results.json")
                with open(all_results_file, 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
            
            best_results = {}
            for key, value in method_results.items():
                if isinstance(value, dict) and 'best' in value and 'best_params' in value:
                    metric_name = key
                    best_params = value['best_params']
                    
                    grid_best_params = {param: best_params[param] 
                                      for param in grid_param_names 
                                      if param in best_params}
                    
                    best_result_entry = None
                    for result_entry in all_results:
                        matches = all(result_entry.get(param) == grid_best_params.get(param) 
                                    for param in grid_param_names)
                        if matches:
                            best_result_entry = result_entry.copy()
                            break
                    
                    # If we have best_predicted_labels in the aggregated results, use those directly
                    if best_result_entry and 'best_predicted_labels' in value:
                        best_result_entry['predicted_labels'] = value['best_predicted_labels']
                    
                    best_results[metric_name] = best_result_entry
            
            if best_results:
                best_results_file = os.path.join(method_dir, f"{method_name}_best_results.json")
                with open(best_results_file, 'w') as f:
                    json.dump(best_results, f, indent=2, default=str)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = os.path.join(dataset_dir, f"{dataset_name}_summary.csv")
            summary_df.to_csv(summary_file, index=False)
    
