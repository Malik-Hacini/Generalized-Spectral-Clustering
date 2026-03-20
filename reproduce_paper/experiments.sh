#!/usr/bin/env bash

run_paper_experiments() {
  section "Experiments"

  run_experiment \
    "UCI benchmark" \
    "experiments/uci.py" \
    "results/benchmark_uci_grid_search/benchmark_uci_params.json"

  run_experiment \
    "Networks benchmark" \
    "experiments/networks.py" \
    "results/networks_grid_search/networks_params.json"

  run_experiment \
    "Grid imbalance benchmark" \
    "experiments/grid_imbalance.py" \
    "results/benchmark_grid_imbalance_grid_search/benchmark_grid_imbalance_params.json"

  run_experiment \
    "Gaussian injection benchmark" \
    "experiments/gaussian_injection_alpha_sigma.py" \
    "results/benchmark_gaussian_injection_alpha_sigma_grid_search/benchmark_gaussian_injection_alpha_sigma_params.json"

  run_experiment \
    "UCI single-run runtime benchmark" \
    "experiments/uci_single_run.py" \
    "results/benchmark_uci_single_run_grid_search/benchmark_uci_single_run_params.json"

  run_experiment \
    "Runtime-size benchmark" \
    "experiments/runtimes_size.py" \
    "results/benchmark_runtimes_size_grid_search/benchmark_runtimes_size_runtimes.csv" \
    1

  run_experiment \
    "Runtime-size preprocessing benchmark" \
    "experiments/runtimes_size_preprocessing.py" \
    "results/benchmark_runtimes_size_preprocessing/benchmark_runtimes_size_preprocessing_runtimes.csv" \
    1
}
