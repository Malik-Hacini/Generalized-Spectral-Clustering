from inspect import signature


class ExperimentConfig:
    """Configuration class to handle parameter hierarchies for clustering experiments.

    This class manages a four-level parameter hierarchy that allows flexible configuration
    of clustering experiments with support for callable parameters and method-specific
    parameter filtering.

    Parameter Hierarchy (lowest to highest precedence):
    1. default_params: Base parameters applied to all methods and datasets
    2. dataset_params: Override defaults for specific datasets
    3. method_params: Override dataset params for specific methods
    4. method_dataset_params: Override method params for specific method-dataset combinations

    Callable Parameter Support:
    --------------------------
    Parameters can be specified as (callable, dict) tuples for dynamic computation:
    - callable: Function to call at runtime
    - dict: Arguments to pass to the callable function

    The callable will receive context arguments (X, adjacency_matrix) plus the specified dict.

    Parameters:
    -----------
    default_params : dict
        Base parameters applied to all experiments
    dataset_params : list of tuple
        List of (dataset_name, params_dict) tuples for dataset-specific overrides
    method_params : list of tuple
        List of (method_name, params_dict) tuples for method-specific overrides
    method_dataset_params : list of tuple
        List of (method_name, dataset_params_list) tuples where dataset_params_list
        contains (dataset_name, params_dict) tuples for method-dataset combinations

    Examples:
    ---------
    >>> # Simple configuration
    >>> config = ExperimentConfig(
    ...     default_params={"n_clusters": 3, "random_state": 42},
    ...     dataset_params=[("iris", {"n_clusters": 3})],
    ...     method_params=[("Spectral", {"affinity": "rbf"})],
    ...     method_dataset_params=[]
    ... )

    >>> # Configuration with callable parameters
    >>> def adaptive_neighbors(X, multiplier=2, **kwargs):
    ...     return X.shape[0] // multiplier

    >>> config = ExperimentConfig(
    ...     default_params={
    ...         "n_neighbors": (adaptive_neighbors, {"multiplier": 3}),
    ...         "random_state": 42
    ...     },
    ...     dataset_params=[],
    ...     method_params=[],
    ...     method_dataset_params=[]
    ... )

    Methods:
    --------
    get_final_params(dataset_name, method_explicit_name, method_implicit_name=None)
        Get final parameters for a specific dataset-method combination
    """

    def __init__(self, default_params, dataset_params, method_params, method_dataset_params):
        self.default_params = default_params
        self.dataset_params_dict = dict(dataset_params) if dataset_params else {}
        self.method_params_dict = dict(method_params) if method_params else {}

        self.method_dataset_params_dict = {}
        if method_dataset_params:
            for method_name, dataset_param_list in method_dataset_params:
                self.method_dataset_params_dict[method_name] = dict(dataset_param_list)

    def get_final_params(self, labels, dataset_name, method_explicit_name, method_implicit_name=None):
        """Get final parameters with hierarchy: default < dataset < method < method_dataset.

        Applies the parameter hierarchy and optionally filters parameters for specific methods.

        Parameters:
        -----------
        labels : the dataset labels, used to determine the number of clusters
        dataset_name : str
            Name of the dataset
        method_explicit_name : str
            The method name explicitely used in the results
        method_implicit_name : str, optional
            Internal method name for parameter filtering (e.g., "spectral", "kmeans")

        Returns:
        --------
        dict
            Final parameters dictionary with hierarchy applied and method filtering
        """
        params = self.default_params.copy()
        if dataset_name in self.dataset_params_dict:
            params.update(self.dataset_params_dict[dataset_name])

        if method_explicit_name in self.method_params_dict:
            params.update(self.method_params_dict[method_explicit_name])

        if (method_explicit_name in self.method_dataset_params_dict and
            dataset_name in self.method_dataset_params_dict[method_explicit_name]):
            params.update(self.method_dataset_params_dict[method_explicit_name][dataset_name])

        if method_implicit_name:
             params= self._filter_params_for_method(params, method_implicit_name)

        params["n_clusters"] = len(set(labels))
        return params


    def _filter_params_for_method(self, params, method_implicit_name):
        """Filter parameters to only include those used by the method."""
        method_param_mapping = {
            "spectral": {"n_clusters", "n_neighbors", "affinity", "laplacian_method",
                        "measure", "random_state", "callable_kwargs", "standard","eigen_solver","eigen_tol" "n_it"},
            "kmeans": {"n_clusters", "random_state", "n_it"},

            "dsc": {"n_clusters", "n_neighbors", "gamma", "max_iter", "tol", "epsilon", "random_state", "n_it"},
            "di_sim": {"n_clusters", "n_neighbors", "tau", "embedding", "epsilon", "random_state", "n_it"},
        }

        if method_implicit_name in method_param_mapping:
            valid_params = method_param_mapping[method_implicit_name]
            return {k: v for k, v in params.items() if k in valid_params}

        return params
