"""Plot runtime curves and theory deviations from benchmark summary CSV files."""
# pyright: reportGeneralTypeIssues=false

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES_DIR, RESULTS_DIR, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
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


def default_inputs() -> list[Path]:
    return sorted((RESULTS_DIR).glob("*_summary.csv"))


def scaled_theory(y: np.ndarray, theory: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(theory) & (y > 0.0) & (theory > 0.0)
    if mask.sum() == 0:
        return 1.0
    return float(np.median(y[mask] / theory[mask]))


def make_figure(df: pd.DataFrame, title: str, out_stem: Path) -> None:
    theory_values = (
        np.asarray(pd.to_numeric(df["theory_term"], errors="coerce"), dtype=float)
        if "theory_term" in df.columns
        else np.array([], dtype=float)
    )
    has_theory = theory_values.size > 0 and np.isfinite(theory_values).any()

    if has_theory:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
        ax_time, ax_ratio = axes
    else:
        fig, ax_time = plt.subplots(1, 1, figsize=(5.4, 4.2), constrained_layout=True)
        ax_ratio = None

    for series, group in df.groupby("series", sort=False):
        group = group.sort_values("x_value")
        x = group["x_value"].to_numpy(dtype=float)
        y = group["runtime_mean"].to_numpy(dtype=float)
        yerr = group["runtime_std"].to_numpy(dtype=float)

        line = ax_time.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=3, label=str(series))
        color = line[0].get_color()

        theory = (
            np.asarray(pd.to_numeric(group["theory_term"], errors="coerce"), dtype=float)
            if "theory_term" in group.columns
            else np.full_like(y, np.nan)
        )
        if has_theory and np.isfinite(theory).any():
            factor = scaled_theory(y, theory)
            label = str(group["theory_label"].iloc[0])
            ax_time.plot(x, factor * theory, linestyle="--", linewidth=1.2, color=color, alpha=0.9, label=f"{series} theory: {label}")
            if ax_ratio is not None:
                ratio = np.where(theory > 0.0, y / theory, np.nan)
                ax_ratio.plot(x, ratio, marker="o", linewidth=1.5, color=color, label=str(series))
                finite = ratio[np.isfinite(ratio)]
                if finite.size:
                    ax_ratio.axhline(float(np.median(finite)), linestyle=":", linewidth=1.0, color=color, alpha=0.7)

    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel(str(df["x_name"].iloc[0]))
    ax_time.set_ylabel("runtime (s)")
    ax_time.set_title(title)
    ax_time.legend(loc="best")

    if ax_ratio is not None:
        ax_ratio.set_xscale("log")
        ax_ratio.set_yscale("log")
        ax_ratio.set_xlabel(str(df["x_name"].iloc[0]))
        ax_ratio.set_ylabel("runtime / theory")
        ax_ratio.set_title("Deviation from theory")
        ax_ratio.legend(loc="best")

    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    style()
    out_dir = ensure_dir(args.out_dir)
    inputs = args.inputs or default_inputs()
    if not inputs:
        raise RuntimeError(f"No summary CSV files found in {RESULTS_DIR}")

    for path in inputs:
        df = pd.read_csv(path)
        title = str(df["benchmark"].iloc[0]).replace("_", " ").title()
        out_stem = out_dir / path.stem.replace("_summary", "")
        make_figure(df, title, out_stem)
        print(f"Saved {out_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
