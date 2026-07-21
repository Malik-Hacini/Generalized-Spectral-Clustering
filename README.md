# Generalized Spectral Clustering

> **This repository is actively maintained thus may evolve in the future. For reproducibility concerns, you will find the exact code used to build the paper upon publication under the paper_state branch.**

Official implementation for the paper [Generalized Dirichlet Energy and Graph Laplacians for Clustering Directed and Undirected Graphs](https://arxiv.org/abs/2203.03221) by Harry Sevi, Gwendal Debaussart-Joenic, Malik Hacini, Matthieu Jonckheere and Argyris Kalogeratos, TMLR 2026.


It provides a robust Python implementation of the Generalized Spectral Clustering (GSC) algorithm via a custom scikit-learn fork. This implementation thus uses SOTA algorithms and is optimized for parallel computing on CPU via JIT compilation.

It also provides a flexible framework for custom clustering experiments.
The exact experiments and plots/figures included in the paper can be reproduced via the `reproduce_paper.sh` shell script. For more information on this tool, see the associated [README](reproduce_paper/README.md).

### GSC Implementation

The scikit-learn fork is identical to the 1.8 version of scikit-learn except for the SpectralClustering class and related utilities.

Two parameters are added to the API call for this class: `standard` (bool) and `laplacian_method` (str: `unnorm`, `norm`, `random_walk`).

If `standard` is True, the adjacency matrix is symmetrized and the standard Laplacian is computed according to the specified method.
Otherwise, GSC is performed, using the specified laplacian method.


### Setting up the experimental environment

Create a virtual environment and activate it (conda or venv):
```
python -m venv sklearn-env
source sklearn-env/bin/activate.fish
```

Install all required dependencies:
```
pip install -r requirements.txt
```

Install scikit-learn in editable mode:
```
pip install -e scikit-learn \
          --no-build-isolation \
          --config-settings editable-verbose=true \
          --verbose
```

## Experiments

### Datasets

Point Cloud datasets are stored in Hugging Face format and network datasets are stored in `.npz` format via their adjacency matrix.
All file manipulations are handled via the [file manager script](utils/file_manager.py).
This pipeline also includes an interactive 2D labeled dataset builder. It is accessible via the `build_dataset()` function in `utils/dataset_builder.py`.

### Pipeline

You can edit and run experiments using [uci.py](experiments/uci.py) as a reference.
An "experiment" consists of datasets to cluster with given methods and fully customizable parameters.
The clusterings are then evaluated using the metrics of your choice (currently available: `nmi`, `ari`, `ami`, `ch`, `modularity`, `graph_ch`).
Three experiment pipelines are available:
- **Score**: Saves results as CSV double-entry tables for each metric
- **Visualization**: Visualizes results in a matplotlib plot, for 2D point-cloud datasets.
- **Grid Search**: Perform a classical score experiment with grid search on specified parameters.

### Adding Custom Components
Detailed guidelines for adding custom components are included in the [competitors](competitors/) module.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
