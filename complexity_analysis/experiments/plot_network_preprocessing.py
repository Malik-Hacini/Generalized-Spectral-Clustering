"""Plot SC and GSC preprocessing costs on sparse DISBM networks."""

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
        default=RESULTS_DIR / "network_preprocessing_disbm_summary.csv",
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
    ax_time, ax_ratio = axes

    for series, group in df.groupby("series", sort=False):
        x = group["nnz"].to_numpy(dtype=float)
        y = group["runtime_mean"].to_numpy(dtype=float)
        yerr = group["runtime_std"].to_numpy(dtype=float)
        theory = group["theory_term"].to_numpy(dtype=float)
        factor = float(np.median(y / theory)) if np.all(theory > 0.0) else 1.0
        line = ax_time.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.7, capsize=3, label=str(series))
        color = line[0].get_color()
        ax_time.plot(x, factor * theory, linestyle="--", linewidth=1.2, color=color, alpha=0.85)
        ax_ratio.plot(x, y / theory, marker="o", linewidth=1.7, color=color, label=str(series))

    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("m = nnz(W)")
    ax_time.set_ylabel("runtime (s)")
    ax_time.set_title("SC vs GSC preprocessing")
    ax_time.legend(loc="best")

    ax_ratio.set_xscale("log")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlabel("m = nnz(W)")
    ax_ratio.set_ylabel("runtime / m")
    ax_ratio.set_title("Deviation from linear sparse scaling")
    ax_ratio.legend(loc="best")

    out_stem = out_dir / "network_preprocessing_sc_vs_gsc"
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
