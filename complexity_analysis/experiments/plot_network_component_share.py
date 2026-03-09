"""Plot preprocessing-to-eigensolver dominance ratios on sparse DISBM networks."""

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
        default=RESULTS_DIR / "network_component_share_disbm_summary.csv",
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

    x = df["nnz"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ax_abs, ax_ratio = axes

    ax_abs.plot(x, df["sc_eigensolver_mean"], marker="o", linewidth=1.7, label="SC eigensolver")
    ax_abs.plot(x, df["gsc_eigensolver_mean"], marker="o", linewidth=1.7, label="GSC eigensolver")
    ax_abs.plot(x, df["sc_preprocessing_mean"], marker="o", linewidth=1.7, label="SC preprocessing")
    ax_abs.plot(x, df["gsc_preprocessing_mean"], marker="o", linewidth=1.7, label="GSC preprocessing")
    ax_abs.set_xscale("log")
    ax_abs.set_yscale("log")
    ax_abs.set_xlabel("m = nnz(W)")
    ax_abs.set_ylabel("runtime (s)")
    ax_abs.set_title("Component magnitudes")
    ax_abs.legend(loc="best")

    ax_ratio.errorbar(x, df["sc_pre_over_eig_mean"], yerr=df["sc_pre_over_eig_std"], marker="o", linewidth=1.7, capsize=3, label="SC preprocessing / eig")
    ax_ratio.errorbar(x, df["gsc_pre_over_eig_mean"], yerr=df["gsc_pre_over_eig_std"], marker="o", linewidth=1.7, capsize=3, label="GSC preprocessing / eig")
    ax_ratio.errorbar(x, df["extra_over_gsc_eig_mean"], yerr=df["extra_over_gsc_eig_std"], marker="o", linewidth=1.7, capsize=3, label="Extra GSC overhead / GSC eig")
    ax_ratio.set_xscale("log")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlabel("m = nnz(W)")
    ax_ratio.set_ylabel("relative contribution")
    ax_ratio.set_title("Preprocessing versus spectral step")
    ax_ratio.legend(loc="best")

    out_stem = out_dir / "network_component_share"
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
