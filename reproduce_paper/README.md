# Paper Reproduction Pipeline

This folder contains the modular shell scripts used to reproduce the experiments and assets for the paper from scratch. The main entry point is the `reproduce_paper.sh` script located in the repository root.

## Architecture

The pipeline is split into logical modules to keep it clean and maintainable:

- `main.sh`: The main runner. It parses command-line arguments, activates either a Python virtual environment or a conda environment, sources the other modules, and executes the pipeline steps in order.
- `common.sh`: Shared utilities and configuration. It defines paths, dataset lists, logging functions, failure tracking, and the experiment reuse logic.
- `experiments.sh`: Defines the `run_paper_experiments` function, which lists and executes all required benchmark scripts.
- `plots.sh`: Defines the `run_paper_plots` function, which generates all figures and tables, and copies them to the configured paper directory's `figures/` and `tables/` directories.

The paper consumes the generated tables directly from:

- `tables/uci/competitors.tex`
- `tables/networks/competitors.tex`
- `tables/measures/uci.tex`
- `tables/measures/networks.tex`
- `tables/dataset_stats.tex`

## Usage

You should normally run the pipeline from the repository root using the wrapper script:

```bash
./reproduce_paper.sh
```

### Options

The script accepts several options to control execution and caching:

- `--venv <NAME_OR_PATH>`: Specify the Python virtual environment to use. Defaults to `.venv` in the repository root.
- `--conda <NAME_OR_PREFIX>`: Specify a conda environment by name or prefix path.
- `--reuse-results`: Skip running benchmark experiments if their main results file already exists. Useful for tweaking plots without waiting for long benchmarks.
- `--reuse-long-results`: Skip only the especially slow experiments (like `runtimes_size.py`) if their results exist, while still re-running the standard benchmarks.
- `--paper-dir <PATH>`: Write generated figures and tables into this paper repo/directory. Defaults to `gsc-tmlr/` when that nested repo exists, otherwise `latex/`.

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

Use a conda environment:
```bash
./reproduce_paper.sh --conda my_conda_env
```

Write assets to a specific paper checkout:
```bash
./reproduce_paper.sh --reuse-results --paper-dir gsc-tmlr
```
