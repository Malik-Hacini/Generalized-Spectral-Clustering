#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/reproduce_paper"
if [[ -n "${PAPER_DIR:-}" ]]; then
  if [[ "$PAPER_DIR" = /* ]]; then
    LATEX_DIR="$PAPER_DIR"
  else
    LATEX_DIR="$ROOT_DIR/$PAPER_DIR"
  fi
elif [[ -d "$ROOT_DIR/gsc-tmlr/.git" ]]; then
  LATEX_DIR="$ROOT_DIR/gsc-tmlr"
else
  LATEX_DIR="$ROOT_DIR/latex"
fi
LATEX_FIGURES="$LATEX_DIR/figures"
LATEX_TABLES="$LATEX_DIR/tables"
TMP_DIR="$ROOT_DIR/.paper_build"

UCI_DATASETS=(breast_tissue wine iris seeds segmentation wdbc olivetti_faces mnist64 ph_recognition)
UCI_HEATMAP_DATASETS=(iris wine wdbc breast_tissue seeds segmentation mnist64 olivetti_faces ph_recognition)
NETWORK_HEATMAP_DATASETS=(DiSBM_Chain Deg-corr football polbooks polblogs email_eu_core)
INDEGREE_DATASETS=("${UCI_DATASETS[@]}")
DATASET_INFO_DATASETS=(DiSBM_Chain Deg-corr email_eu_core football polblogs polbooks breast_tissue iris mnist64 olivetti_faces ph_recognition seeds segmentation wdbc wine)
EXPECTED_GRID_IMBALANCE_FILES=(grid_imbalance_2x1.pdf grid_imbalance_2x2.pdf grid_imbalance_3x3.pdf grid_imbalance_4x4.pdf)

format_seconds() {
  local total="$1"
  local mins=$(( total / 60 ))
  local secs=$(( total % 60 ))
  printf "%dm%02ds" "$mins" "$secs"
}

section() {
  printf '\n%s\n' "============================================================"
  printf '%s\n' "$1"
  printf '%s\n' "============================================================"
}

warn() {
  printf '[WARN] %s\n' "$1" >&2
}

record_failure() {
  FAILURES+=("$1")
  warn "$1"
}

run_timed() {
  local label="$1"
  shift
  local start=$SECONDS

  printf '\n[RUN] %s\n' "$label"
  if "$@"; then
    local elapsed=$(( SECONDS - start ))
    printf '[ OK ] %s (%s)\n' "$label" "$(format_seconds "$elapsed")"
  else
    local status=$?
    local elapsed=$(( SECONDS - start ))
    printf '[FAIL] %s (%s, exit=%d)\n' "$label" "$(format_seconds "$elapsed")" "$status" >&2
    record_failure "$label"
  fi
}

should_reuse() {
  local sentinel="$1"
  local is_long="$2"

  if [[ ! -e "$ROOT_DIR/$sentinel" ]]; then
    return 1
  fi
  if [[ "$REUSE_RESULTS" -eq 1 ]]; then
    return 0
  fi
  if [[ "$is_long" -eq 1 && "$REUSE_LONG_RESULTS" -eq 1 ]]; then
    return 0
  fi
  return 1
}

run_experiment() {
  local label="$1"
  local script_path="$2"
  local sentinel="$3"
  local is_long=0
  if [[ $# -ge 4 ]]; then
    is_long="$4"
    shift 4
  else
    shift 3
  fi

  if should_reuse "$sentinel" "$is_long"; then
    printf '\n[SKIP] %s (reusing %s)\n' "$label" "$sentinel"
    return
  fi

  run_timed "$label" "$PYTHON_BIN" "$ROOT_DIR/$script_path" "$@"
}

copy_file() {
  local src="$1"
  local dst="$2"

  if [[ ! -f "$src" ]]; then
    record_failure "Missing asset: $src"
    return 1
  fi

  mkdir -p "$(dirname "$dst")"
  if ! cp "$src" "$dst"; then
    record_failure "Failed to copy $src -> $dst"
    return 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    record_failure "Missing expected output: $path"
    return 1
  fi
}

prepare_environment() {
  mkdir -p "$LATEX_FIGURES" "$LATEX_TABLES" "$TMP_DIR"
}

print_environment() {
  section "Environment"
  printf 'Root directory : %s\n' "$ROOT_DIR"
  printf '%-14s %s\n' "${ENV_LABEL:-Python env}:" "$VENV_DIR"
  printf 'Python         : %s\n' "$PYTHON_BIN"
  printf 'Paper dir      : %s\n' "$LATEX_DIR"
  printf 'Paper figures  : %s\n' "$LATEX_FIGURES"
  printf 'Paper tables   : %s\n' "$LATEX_TABLES"
}

print_summary() {
  section "Summary"
  local status=0
  if [[ ${#FAILURES[@]} -eq 0 ]]; then
    printf '[ OK ] Reproduction pipeline completed without reported failures.\n'
  else
    printf '[WARN] Reproduction pipeline completed with %d issue(s):\n' "${#FAILURES[@]}"
    for failure in "${FAILURES[@]}"; do
      printf '  - %s\n' "$failure"
    done
    status=1
  fi

  printf 'Paper assets are available under %s and %s\n' "$LATEX_FIGURES" "$LATEX_TABLES"
  printf 'You can compile the paper separately, e.g. with: (cd "%s" && latexmk -pdf main.tex)\n' "$LATEX_DIR"
  return "$status"
}
