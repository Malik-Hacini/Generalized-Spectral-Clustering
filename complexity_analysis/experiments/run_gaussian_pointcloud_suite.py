"""Run the recommended Gaussian point-cloud complexity benchmark suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENTS_DIR.parents[1]
DATA_DIR = EXPERIMENTS_DIR / "data"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-n-values", default="500,1000,2000,5000,10000")
    parser.add_argument("--component-n-values", default="500,1000,2000,5000,10000")
    parser.add_argument("--full-n-values", default="500,1000,2000,5000")
    parser.add_argument("--graph-dimension", type=int, default=4)
    parser.add_argument("--baseline-dimension", type=int, default=10)
    parser.add_argument("--dimension-sweep-n", type=int, default=5000)
    parser.add_argument("--dimension-sweep-values", default="2,4,8,12,16,24,32,48")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--cluster-std", type=float, default=2.5)
    parser.add_argument("--neighbors-factor", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--component-repeats", type=int, default=5)
    parser.add_argument("--eigensolver-repeats", type=int, default=3)
    parser.add_argument("--full-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def merge_csv_lists(*lists: str) -> str:
    seen: set[int] = set()
    out: list[str] = []
    for text in lists:
        for chunk in text.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            value = int(chunk)
            if value in seen:
                continue
            seen.add(value)
            out.append(str(value))
    return ",".join(out)


def command(script: str, *args: str, python: str) -> list[str]:
    return [python, str(EXPERIMENTS_DIR / script), *args]


def build_suite(args: argparse.Namespace) -> tuple[list[list[str]], list[Path]]:
    commands: list[list[str]] = []
    expected_summaries: list[Path] = []

    if not args.skip_generate:
        dataset_ns = merge_csv_lists(
            args.graph_n_values,
            args.component_n_values,
            args.full_n_values,
            str(args.dimension_sweep_n),
        )
        dataset_ds = merge_csv_lists(
            str(args.graph_dimension),
            str(args.baseline_dimension),
            args.dimension_sweep_values,
        )
        commands.append(
            command(
                "generate_gaussian_pointcloud.py",
                "--n-values",
                dataset_ns,
                "--d-values",
                dataset_ds,
                "--clusters",
                str(args.clusters),
                "--cluster-std",
                str(args.cluster_std),
                "--neighbors-factor",
                str(args.neighbors_factor),
                "--seed",
                str(args.seed),
                "--out-dir",
                str(DATA_DIR),
                python=args.python,
            )
        )

    benchmark_specs = [
        (
            "graph_construction_n_summary.csv",
            command(
                "benchmark_graph_construction.py",
                "--sweep",
                "n",
                "--n-values",
                args.graph_n_values,
                "--dimension",
                str(args.graph_dimension),
                "--algorithm",
                "auto",
                "--clusters",
                str(args.clusters),
                "--cluster-std",
                str(args.cluster_std),
                "--neighbors-factor",
                str(args.neighbors_factor),
                "--repeats",
                str(args.component_repeats),
                "--seed",
                str(args.seed),
                "--n-jobs",
                str(args.n_jobs),
                "--out-dir",
                str(RESULTS_DIR),
                python=args.python,
            ),
        ),
        (
            "graph_construction_d_summary.csv",
            command(
                "benchmark_graph_construction.py",
                "--sweep",
                "d",
                "--n-samples",
                str(args.dimension_sweep_n),
                "--d-values",
                args.dimension_sweep_values,
                "--algorithm",
                "auto",
                "--clusters",
                str(args.clusters),
                "--cluster-std",
                str(args.cluster_std),
                "--neighbors-factor",
                str(args.neighbors_factor),
                "--repeats",
                str(args.component_repeats),
                "--seed",
                str(args.seed),
                "--n-jobs",
                str(args.n_jobs),
                "--out-dir",
                str(RESULTS_DIR),
                python=args.python,
            ),
        ),
        (
            "measure_construction_summary.csv",
            command(
                "benchmark_measure_construction.py",
                "--n-values",
                args.component_n_values,
                "--dimension",
                str(args.baseline_dimension),
                "--clusters",
                str(args.clusters),
                "--cluster-std",
                str(args.cluster_std),
                "--neighbors-factor",
                str(args.neighbors_factor),
                "--alpha",
                str(args.alpha),
                "--t",
                str(args.t),
                "--repeats",
                str(args.component_repeats),
                "--seed",
                str(args.seed),
                "--n-jobs",
                str(args.n_jobs),
                "--out-dir",
                str(RESULTS_DIR),
                python=args.python,
            ),
        ),
        (
            "xi_construction_summary.csv",
            command(
                "benchmark_xi_construction.py",
                "--n-values",
                args.component_n_values,
                "--dimension",
                str(args.baseline_dimension),
                "--clusters",
                str(args.clusters),
                "--cluster-std",
                str(args.cluster_std),
                "--neighbors-factor",
                str(args.neighbors_factor),
                "--alpha",
                str(args.alpha),
                "--t",
                str(args.t),
                "--repeats",
                str(args.component_repeats),
                "--seed",
                str(args.seed),
                "--n-jobs",
                str(args.n_jobs),
                "--out-dir",
                str(RESULTS_DIR),
                python=args.python,
            ),
        ),
        (
            "laplacian_construction_summary.csv",
            command(
                "benchmark_laplacian_construction.py",
                "--n-values",
                args.component_n_values,
                "--dimension",
                str(args.baseline_dimension),
                "--clusters",
                str(args.clusters),
                "--cluster-std",
                str(args.cluster_std),
                "--neighbors-factor",
                str(args.neighbors_factor),
                "--alpha",
                str(args.alpha),
                "--t",
                str(args.t),
                "--repeats",
                str(args.component_repeats),
                "--seed",
                str(args.seed),
                "--n-jobs",
                str(args.n_jobs),
                "--out-dir",
                str(RESULTS_DIR),
                python=args.python,
            ),
        ),
        (
            "eigensolver_pointcloud_summary.csv",
            command(
                "benchmark_eigensolver.py",
                "--input-type",
                "pointcloud",
                "--n-values",
                args.full_n_values,
                "--dimension",
                str(args.baseline_dimension),
                "--clusters",
                str(args.clusters),
                "--n-components",
                str(args.n_components),
                "--cluster-std",
                str(args.cluster_std),
                "--neighbors-factor",
                str(args.neighbors_factor),
                "--alpha",
                str(args.alpha),
                "--t",
                str(args.t),
                "--repeats",
                str(args.eigensolver_repeats),
                "--seed",
                str(args.seed),
                "--n-jobs",
                str(args.n_jobs),
                "--out-dir",
                str(RESULTS_DIR),
                python=args.python,
            ),
        ),
        (
            "full_run_pointcloud_summary.csv",
            command(
                "benchmark_full_run.py",
                "--input-type",
                "pointcloud",
                "--n-values",
                args.full_n_values,
                "--dimension",
                str(args.baseline_dimension),
                "--clusters",
                str(args.clusters),
                "--cluster-std",
                str(args.cluster_std),
                "--neighbors-factor",
                str(args.neighbors_factor),
                "--alpha",
                str(args.alpha),
                "--t",
                str(args.t),
                "--repeats",
                str(args.full_repeats),
                "--seed",
                str(args.seed),
                "--out-dir",
                str(RESULTS_DIR),
                python=args.python,
            ),
        ),
    ]

    for name, cmd in benchmark_specs:
        commands.append(cmd)
        path = RESULTS_DIR / name
        if path not in expected_summaries:
            expected_summaries.append(path)

    if not args.skip_plots:
        plot_args = [str(path) for path in expected_summaries]
        commands.append(
            command(
                "plot_benchmarks.py",
                *plot_args,
                "--out-dir",
                str(FIGURES_DIR),
                python=args.python,
            )
        )

    return commands, expected_summaries


def print_command(cmd: list[str]) -> None:
    print("$", " ".join(f'"{part}"' if " " in part else part for part in cmd))


def main() -> None:
    args = parse_args()
    commands, summaries = build_suite(args)

    print("Running Gaussian point-cloud complexity suite")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")

    for cmd in commands:
        print_command(cmd)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=ROOT, check=True)

    print("Done.")
    print("Expected summary files:")
    for path in summaries:
        print(f"- {path}")


if __name__ == "__main__":
    main()
