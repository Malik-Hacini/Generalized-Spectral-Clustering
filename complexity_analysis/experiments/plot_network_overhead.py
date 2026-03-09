"""Plot the GSC overhead contribution versus sparse edge count m."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import FIGURES_DIR, RESULTS_DIR, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=RESULTS_DIR / "network_overhead_disbm_components_summary.csv",
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
    df = pd.read_csv(summary_csv).sort_values("nnz")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ax_abs, ax_frac = axes

    x = df["nnz"].to_numpy(dtype=float)
    overhead = df["extra_overhead_mean"].to_numpy(dtype=float)
    overhead_std = df["extra_overhead_std"].to_numpy(dtype=float)
    frac = df["overhead_fraction_of_gsc_mean"].to_numpy(dtype=float)
    frac_std = df["overhead_fraction_of_gsc_std"].to_numpy(dtype=float)

    ax_abs.errorbar(x, overhead, yerr=overhead_std, marker="o", linewidth=1.8, capsize=3, color="#1f77b4")
    ax_abs.set_xscale("log")
    ax_abs.set_yscale("log")
    ax_abs.set_xlabel("m = nnz(W)")
    ax_abs.set_ylabel("extra GSC overhead (s)")
    ax_abs.set_title("Absolute preprocessing overhead")

    ax_frac.errorbar(x, frac, yerr=frac_std, marker="o", linewidth=1.8, capsize=3, color="#d62728")
    ax_frac.set_xscale("log")
    ax_frac.set_yscale("log")
    ax_frac.set_xlabel("m = nnz(W)")
    ax_frac.set_ylabel("extra overhead / total GSC runtime")
    ax_frac.set_title("Relative overhead contribution")

    out_stem = out_dir / "network_gsc_overhead_vs_m"
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
