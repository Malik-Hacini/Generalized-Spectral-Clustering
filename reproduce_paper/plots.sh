#!/usr/bin/env bash

run_intro_figures() {
  section "Paper assets: introductory figures"

  run_timed \
    "Generate introductory Dirichlet/GDE figures" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/dirichlet.py" \
    --output-dir "$LATEX_FIGURES"

  require_file "$LATEX_FIGURES/clustering_ergodic.pdf"
  require_file "$LATEX_FIGURES/dirichlet_true_labels.pdf"
  require_file "$LATEX_FIGURES/dirichlet_mixed_labels.pdf"
}

run_heatmaps() {
  section "Paper assets: GSC-N sensitivity heatmaps"

  local uci_tmp="$TMP_DIR/heatmaps_uci"
  local network_tmp="$TMP_DIR/heatmaps_networks"

  run_timed \
    "Generate UCI GSC-N heatmaps" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/heatmaps_gsc.py" \
    --results-dir results/benchmark_uci_grid_search \
    --metrics CH AMI \
    --datasets "${UCI_HEATMAP_DATASETS[@]}" \
    --methods GSC-N \
    --output-dir "$uci_tmp"

  for dataset in "${UCI_HEATMAP_DATASETS[@]}"; do
    copy_file "$uci_tmp/CH/${dataset}_GSC-N_t_alpha_ch.pdf" "$LATEX_FIGURES/${dataset}_GSC-N_t_alpha_ch.pdf"
    copy_file "$uci_tmp/AMI/${dataset}_GSC-N_t_alpha_ami.pdf" "$LATEX_FIGURES/${dataset}_GSC-N_t_alpha_ami.pdf"
  done

  run_timed \
    "Generate network GSC-N heatmaps" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/heatmaps_gsc.py" \
    --results-dir results/networks_grid_search \
    --metrics GRAPH_CH AMI \
    --datasets "${NETWORK_HEATMAP_DATASETS[@]}" \
    --methods GSC-N \
    --output-dir "$network_tmp"

  for dataset in "${NETWORK_HEATMAP_DATASETS[@]}"; do
    copy_file "$network_tmp/GRAPH_CH/${dataset}_GSC-N_t_alpha_graph_ch.pdf" "$LATEX_FIGURES/${dataset}_GSC-N_t_alpha_ch.pdf"
    copy_file "$network_tmp/AMI/${dataset}_GSC-N_t_alpha_ami.pdf" "$LATEX_FIGURES/${dataset}_GSC-N_t_alpha_ami.pdf"
  done
}

run_tables() {
  section "Paper assets: tables"

  run_timed \
    "Generate UCI competitors table" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/tables_competitors.py" \
    --results-dir results/benchmark_uci_grid_search \
    --output-dir "$LATEX_TABLES/uci" \
    --output-name competitors.tex \
    --paper-table uci \
    --datasets breast_tissue iris mnist64 olivetti_faces ph_recognition seeds segmentation wdbc wine

  run_timed \
    "Generate network competitors table" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/tables_competitors.py" \
    --results-dir results/networks_grid_search \
    --output-dir "$LATEX_TABLES/networks" \
    --output-name competitors.tex \
    --paper-table network \
    --datasets Deg-corr DiSBM_Chain email_eu_core football polblogs polbooks

  run_timed \
    "Generate UCI vertex-measure table" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/tables_measures.py" \
    --results-dir results/benchmark_uci_grid_search \
    --output-dir "$LATEX_TABLES/measures" \
    --output-name uci.tex \
    --paper-table uci \
    --datasets breast_tissue iris mnist64 olivetti_faces ph_recognition seeds segmentation wdbc wine

  run_timed \
    "Generate network vertex-measure table" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/tables_measures.py" \
    --results-dir results/networks_grid_search \
    --output-dir "$LATEX_TABLES/measures" \
    --output-name networks.tex \
    --paper-table network \
    --datasets DiSBM_Chain Deg-corr email_eu_core football polblogs polbooks

  run_timed \
    "Generate dataset statistics table" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/tables_dataset_infos.py" \
    --datasets "${DATASET_INFO_DATASETS[@]}" \
    --output-dir "$LATEX_TABLES" \
    --output-name dataset_stats.tex \
    --strict

  require_file "$LATEX_TABLES/uci/competitors.tex"
  require_file "$LATEX_TABLES/networks/competitors.tex"
  require_file "$LATEX_TABLES/measures/uci.tex"
  require_file "$LATEX_TABLES/measures/networks.tex"
  require_file "$LATEX_TABLES/dataset_stats.tex"
}

run_imbalance_and_injection_figures() {
  section "Paper assets: size-imbalance figure"

  run_timed \
    "Generate grid-imbalance figure suite" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/imbalance_grid.py" \
    --results-dir results/benchmark_grid_imbalance_grid_search \
    --output-dir "$LATEX_FIGURES"

  for filename in "${EXPECTED_GRID_IMBALANCE_FILES[@]}"; do
    require_file "$LATEX_FIGURES/$filename"
  done

  section "Paper assets: chain-flow figure"

  run_timed \
    "Generate chain-flow figure" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/curves_chain_flow.py" \
    --results-dir results/benchmark_chain_flow_grid_search \
    --output-dir "$LATEX_FIGURES" \
    --x p_forward

  require_file "$LATEX_FIGURES/chain_flow_ami_vs_p_forward.pdf"

  section "Paper assets: degree-imbalance figures"

  run_timed \
    "Generate Gaussian-injection alpha curve (sigma=1.0)" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/imbalance_gaussian_injection.py" \
    --results-dir results/benchmark_gaussian_injection_alpha_sigma_grid_search \
    --output-dir "$LATEX_FIGURES" \
    --optimize-by graph_ch \
    --skip-sigma-plot \
    --fixed-alpha 0.5 \
    --fixed-sigma 1.0

  run_timed \
    "Generate Gaussian-injection alpha curve (sigma=0.8)" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/imbalance_gaussian_injection.py" \
    --results-dir results/benchmark_gaussian_injection_alpha_sigma_grid_search \
    --output-dir "$LATEX_FIGURES" \
    --optimize-by graph_ch \
    --skip-sigma-plot \
    --fixed-alpha 0.5 \
    --fixed-sigma 0.8

  run_timed \
    "Generate Gaussian-injection sigma curve (alpha=0.5)" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/imbalance_gaussian_injection.py" \
    --results-dir results/benchmark_gaussian_injection_alpha_sigma_grid_search \
    --output-dir "$LATEX_FIGURES" \
    --optimize-by graph_ch \
    --skip-alpha-plot \
    --fixed-alpha 0.5

  require_file "$LATEX_FIGURES/gaussian_injection_ami_mean_std_vs_sigma_alpha0.5000_graph_ch.pdf"
  require_file "$LATEX_FIGURES/gaussian_injection_ami_mean_std_vs_alpha_sigma1.0000_graph_ch.pdf"
  require_file "$LATEX_FIGURES/gaussian_injection_ami_mean_std_vs_alpha_sigma0.8000_graph_ch.pdf"
}

run_runtime_figures() {
  section "Paper assets: runtime figures"

  run_timed \
    "Generate UCI runtime comparison" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/runtimes_benchmark.py" \
    --results-csv results/benchmark_uci_single_run_grid_search/benchmark_uci_single_run_runtimes.csv \
    --output-dir "$LATEX_FIGURES" \
    --output-name benchmark_uci_runtime_comparison \
    --title ""

  run_timed \
    "Generate runtime-vs-size figure" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/runtimes_size.py" \
    --results-csv results/benchmark_runtimes_size_grid_search/benchmark_runtimes_size_runtimes.csv \
    --output-dir "$LATEX_FIGURES" \
    --output-name runtimes_size_lines.pdf

  require_file "$LATEX_FIGURES/benchmark_uci_runtime_comparison.pdf"
  require_file "$LATEX_FIGURES/runtimes_size_lines.pdf"
}

run_appendix_figures() {
  section "Paper assets: appendix figures"

  local indegree_tmp="$TMP_DIR/indegree"

  run_timed \
    "Generate in-degree distribution plots" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/figures_indegree_knn.py" \
    --datasets "${INDEGREE_DATASETS[@]}" \
    --factor 1 \
    --output-dir "$indegree_tmp"

  for dataset in "${INDEGREE_DATASETS[@]}"; do
    copy_file "$indegree_tmp/${dataset}_indegree_factor_1.pdf" "$LATEX_FIGURES/${dataset}_indeg_distribution.pdf"
  done
}

run_additional_runtime_assets() {
  section "Additional runtime asset (rebuttal / complexity)"

  run_timed \
    "Generate preprocessing-vs-full runtime figure" \
    "$PYTHON_BIN" "$ROOT_DIR/plots/runtimes_preprocessing_vs_full.py" \
    --full-results-csv results/benchmark_runtimes_size_grid_search/benchmark_runtimes_size_runtimes.csv \
    --preprocessing-results-csv results/benchmark_runtimes_size_preprocessing/benchmark_runtimes_size_preprocessing_runtimes.csv \
    --output-dir "$LATEX_FIGURES" \
    --output-name runtimes_preprocessing_vs_full.pdf

  require_file "$LATEX_FIGURES/runtimes_preprocessing_vs_full.pdf"
}

run_paper_plots() {
  run_intro_figures
  run_heatmaps
  run_tables
  run_imbalance_and_injection_figures
  run_runtime_figures
  run_appendix_figures
  run_additional_runtime_assets
}
