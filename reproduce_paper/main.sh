#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/reproduce_paper"
VENV_NAME=".venv"
REUSE_RESULTS=0
REUSE_LONG_RESULTS=0
FAILURES=()

usage() {
  cat <<'EOF'
Usage: ./reproduce_paper.sh [options]

Reproduce the experiments and paper assets used in latex/main.tex.

Options:
  --venv NAME_OR_PATH         Virtual environment directory (default: .venv)
  --reuse-results             Reuse existing benchmark results when present
  --reuse-long-results        Reuse existing long runtime-size results when present
  -h, --help                  Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_NAME="$2"
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
