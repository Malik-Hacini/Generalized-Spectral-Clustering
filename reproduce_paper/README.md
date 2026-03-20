# Paper Reproduction Pipeline

This folder contains the modular shell scripts used to reproduce the experiments and assets for the paper from scratch. The main entry point is the `reproduce_paper.sh` script located in the repository root.

## Architecture

The pipeline is split into logical modules to keep it clean and maintainable:

- `main.sh`: The main runner. It parses command-line arguments, activates the virtual environment, sources the other modules, and executes the pipeline steps in order.
- `common.sh`: Shared utilities and configuration. It defines paths, dataset lists, logging functions, failure tracking, and the experiment reuse logic.
- `experiments.sh`: Defines the `run_paper_experiments` function, which lists and executes all required benchmark scripts.
- `plots.sh`: Defines the `run_paper_plots` function, which generates all figures and tables, and copies them to the `latex/figures/` and `latex/tables/` directories.

## Usage

You should normally run the pipeline from the repository root using the wrapper script:

```bash
./reproduce_paper.sh
```

### Options

The script accepts several options to control execution and caching:

- `--venv <NAME_OR_PATH>`: Specify the Python virtual environment to use. Defaults to `.venv` in the repository root.
- `--reuse-results`: Skip running benchmark experiments if their main results file already exists. Useful for tweaking plots without waiting for long benchmarks.
- `--reuse-long-results`: Skip only the especially slow experiments (like `runtimes_size.py`) if their results exist, while still re-running the standard benchmarks.

### Examples

Run everything from scratch:
```bash
./reproduce_paper.sh
```

Regenerate all plots from existing data (skip all experiments if data is present):
```bash
./reproduce_paper.sh --reuse-results
```

Run standard benchmarks from scratch, but skip the very slow runtime-size benchmark if its data exists:
```bash
./reproduce_paper.sh --reuse-long-results
```

Use a custom virtual environment:
```bash
./reproduce_paper.sh --venv my_custom_env
```

## Modifying the Pipeline

When updating the paper:

1. **New Experiments:** Add the execution step to `experiments.sh` using the `run_experiment` helper. Provide a clear label, the script path, the expected output file (sentinel), and an optional flag (`1`) if it is a long-running experiment.
2. **New Plots/Tables:** Add the generation step to the appropriate function in `plots.sh` (e.g., `run_tables`, `run_heatmaps`). Use the `run_timed` helper to execute the Python plotting script.
3. **Copying Assets:** If a script generates an asset in a temporary directory or needs it moved to the `latex/` folder, use the `copy_file` helper in `plots.sh`.
4. **Validating Output:** Use the `require_file` helper in `plots.sh` to explicitly check that a critical asset was successfully generated. This ensures the failure tracking system catches missing figures before you try to compile the LaTeX document.

## Failure Handling

The pipeline is designed to be resilient. If a single experiment or plot generation fails, the script will record the failure, print a warning, and continue with the rest of the pipeline. A summary of all failures is printed at the very end.
