#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/reproduce_paper"
VENV_NAME=".venv"
CONDA_ENV=""
USE_CONDA=0
VENV_SPECIFIED=0
REUSE_RESULTS=0
REUSE_LONG_RESULTS=0
FAILURES=()

usage() {
  cat <<'EOF'
Usage: ./reproduce_paper.sh [options]

Reproduce the experiments and paper assets used in latex/main.tex.

Options:
  --venv NAME_OR_PATH         Virtual environment directory (default: .venv)
  --conda NAME_OR_PREFIX      Conda environment name or prefix path
  --reuse-results             Reuse existing benchmark results when present
  --reuse-long-results        Reuse existing long runtime-size results when present
  -h, --help                  Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_NAME="$2"
      VENV_SPECIFIED=1
      shift 2
      ;;
    --conda)
      CONDA_ENV="$2"
      USE_CONDA=1
      shift 2
      ;;
    --reuse-results)
      REUSE_RESULTS=1
      shift
      ;;
    --reuse-long-results)
      REUSE_LONG_RESULTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$USE_CONDA" -eq 1 && "$VENV_SPECIFIED" -eq 1 ]]; then
  echo "Error: use either --venv or --conda, not both." >&2
  exit 1
fi

if [[ "$USE_CONDA" -eq 1 ]]; then
  if [[ -z "$CONDA_ENV" ]]; then
    echo "Error: --conda requires an environment name or prefix." >&2
    exit 1
  fi

  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
  else
    echo "Error: conda executable not found. Cannot activate '$CONDA_ENV'." >&2
    exit 1
  fi

  CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
  if [[ ! -f "$CONDA_SH" ]]; then
    echo "Error: conda activation script not found: $CONDA_SH" >&2
    exit 1
  fi

  # shellcheck source=/dev/null
  source "$CONDA_SH"
  if ! conda activate "$CONDA_ENV"; then
    echo "Error: failed to activate conda environment '$CONDA_ENV'." >&2
    exit 1
  fi

  PYTHON_BIN="$(command -v python)"
  VENV_DIR="${CONDA_PREFIX:-$CONDA_ENV}"
  ENV_LABEL="Conda env"
else
  if [[ "$VENV_NAME" = /* ]]; then
    VENV_DIR="$VENV_NAME"
  else
    VENV_DIR="$ROOT_DIR/$VENV_NAME"
  fi

  PYTHON_BIN="$VENV_DIR/bin/python"
  ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"

  if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
    echo "Error: virtual environment not found: $VENV_DIR" >&2
    exit 1
  fi

  # shellcheck source=/dev/null
  source "$ACTIVATE_SCRIPT"

  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Error: python executable not found in $VENV_DIR" >&2
    exit 1
  fi
  ENV_LABEL="Python env"
fi

cd "$ROOT_DIR"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/experiments.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/plots.sh"

prepare_environment
trap 'rm -rf "$TMP_DIR"' EXIT

print_environment
run_paper_experiments
run_paper_plots
print_summary
