"""Create one clean SC-vs-GSC plot for precomputed network benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES_DIR, RESULTS_DIR, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=RESULTS_DIR / "full_run_disbm_network_summary.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args()


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> None:
    args = parse_args()
    style()
    out_dir = ensure_dir(args.out_dir)
    summary_csv = args.summary_csv
    if not summary_csv.is_absolute():
        summary_csv = Path.cwd() / summary_csv
    df = pd.read_csv(summary_csv).sort_values(["series", "nnz"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ax_time, ax_diff = axes

    for series, group in df.groupby("series", sort=False):
        x = group["nnz"].to_numpy(dtype=float)
        y = group["runtime_mean"].to_numpy(dtype=float)
        yerr = group["runtime_std"].to_numpy(dtype=float)
        ax_time.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.7, capsize=3, label=str(series))

    pivot = df.pivot(index="nnz", columns="series", values="runtime_mean").sort_index()
    diff_ms = np.asarray(1000.0 * (pivot["GSC"] - pivot["SC"]), dtype=float)
    ax_diff.plot(np.asarray(pivot.index, dtype=float), diff_ms, marker="o", linewidth=1.7, color="#2ca02c")
    ax_diff.axhline(0.0, linestyle=":", linewidth=1.0, color="black", alpha=0.8)

    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("m = nnz(W)")
    ax_time.set_ylabel("runtime (s)")
    ax_time.set_title("SC vs GSC on precomputed DISBM networks")
    ax_time.legend(loc="best")

    ax_diff.set_xscale("log")
    ax_diff.set_xlabel("m = nnz(W)")
    ax_diff.set_ylabel("GSC - SC (ms)")
    ax_diff.set_title("End-to-end runtime difference")

    out_stem = out_dir / "network_sc_vs_gsc_single_plot"
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
