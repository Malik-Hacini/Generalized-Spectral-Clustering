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
    "Grid imbalance benchmark (2x2)" \
    "experiments/grid_imbalance.py" \
    "results/benchmark_grid_imbalance_grid_search/grid_2x2_high300_low20_seed0" \
    0 \
    --grid-size 2

  run_experiment \
    "Grid imbalance benchmark (2x1)" \
    "experiments/grid_imbalance.py" \
    "results/benchmark_grid_imbalance_grid_search/grid_2x1_high300_low20_seed0" \
    0 \
    --grid-size 2x1

  run_experiment \
    "Grid imbalance benchmark (3x3)" \
    "experiments/grid_imbalance.py" \
    "results/benchmark_grid_imbalance_grid_search/grid_3x3_high300_low20_seed0" \
    0 \
    --grid-size 3

  run_experiment \
    "Grid imbalance benchmark (4x4)" \
    "experiments/grid_imbalance.py" \
    "results/benchmark_grid_imbalance_grid_search/grid_4x4_high300_low20_seed0" \
    0 \
    --grid-size 4

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
