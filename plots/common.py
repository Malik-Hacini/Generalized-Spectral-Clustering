from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLOTS_ROOT = ROOT / "plots"

DEFAULT_METHOD_ORDER = [
    "SC-UN",
    "SC-N",
    "DSC+",
    "Chung",
    "DI-SIM-R",
    "DI-SIM-L",
    "DI-SIM-C",
    "GSC-UN",
    "GSC-N",
    "GSC-UN-NoTune",
]

METHOD_STYLES: dict[str, dict[str, str]] = {
    "SC-UN": {"color": "#FF8C69", "linestyle": ":", "marker": "D", "label": "SC-UN"},
    "SC-N": {"color": "#FF6347", "linestyle": "--", "marker": "o", "label": "SC-N"},
    "DSC+": {"color": "#27A727", "linestyle": "-.", "marker": "^", "label": "DSC+"},
    "Chung": {"color": "#008F00", "linestyle": ":", "marker": "v", "label": "Chung"},
    "DI-SIM-R": {"color": "#7A3E9D", "linestyle": "-", "marker": "P", "label": "DI-SIM-R"},
    "DI-SIM-L": {"color": "#A55CC2", "linestyle": "--", "marker": "X", "label": "DI-SIM-L"},
    "DI-SIM-C": {"color": "#C084D8", "linestyle": "-.", "marker": "*", "label": "DI-SIM-C"},
    "GSC-UN": {"color": "#4C9AFF", "linestyle": "--", "marker": "P", "label": "GSC-UN"},
    "GSC-N": {"color": "#072AC8", "linestyle": "-", "marker": "s", "label": "GSC-N"},
    "GSC-UN-NoTune": {"color": "#264DF7", "linestyle": "-.", "marker": "^", "label": "GSC-UN (w/o tuning)"},
}

FALLBACK_COLORS = ["#6C757D", "#8C564B", "#BCBD22", "#17BECF"]
FALLBACK_LINESTYLES = ["-", "--", "-.", ":"]
FALLBACK_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "<", ">", "h", "*"]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def experiment_name(source: str | Path) -> str:
    source_path = project_path(source)
    return source_path.name if source_path.is_dir() else source_path.parent.name


def resolve_output_dir(output_dir: str | Path | None, kind: str, source: str | Path) -> Path:
    output_path = PLOTS_ROOT / kind / experiment_name(source) if output_dir is None else project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def resolve_kind_dir(output_dir: str | Path | None, kind: str) -> Path:
    output_path = PLOTS_ROOT / kind if output_dir is None else project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def resolve_output_file(
    output_dir: str | Path | None,
    output_name: str | None,
    kind: str,
    source: str | Path,
    default_name: str,
) -> Path:
    return resolve_output_dir(output_dir, kind, source) / (output_name or default_name)


def validate_selection(available: list[str], selected: list[str] | None, label: str) -> list[str]:
    if selected is None:
        return available
    missing = [item for item in selected if item not in available]
    if missing:
        raise ValueError(f"Unknown {label}: {missing}. Available {label}: {available}")
    return selected


def ordered_methods(available_methods: Iterable[str], preferred_order: list[str] | None = None) -> list[str]:
    preferred = DEFAULT_METHOD_ORDER if preferred_order is None else preferred_order
    available = list(dict.fromkeys(available_methods))
    selected = [method for method in preferred if method in available]
    return selected + sorted(method for method in available if method not in selected)


def style_for_method(method: str) -> dict[str, str]:
    style = METHOD_STYLES.get(method)
    if style is not None:
        return dict(style)

    fallback_idx = sum(ord(ch) for ch in method)
    return {
        "color": FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)],
        "linestyle": FALLBACK_LINESTYLES[fallback_idx % len(FALLBACK_LINESTYLES)],
        "marker": FALLBACK_MARKERS[fallback_idx % len(FALLBACK_MARKERS)],
        "label": method,
    }


def styles_for_methods(methods: Iterable[str]) -> dict[str, dict[str, str]]:
    return {method: style_for_method(method) for method in methods}


def summarize_mean_std(df: pd.DataFrame, group_cols: list[str], value_col: str, prefix: str | None = None) -> pd.DataFrame:
    summary = df.groupby(group_cols)[value_col].agg(["mean", "std", "count"]).reset_index()
    label = value_col if prefix is None else prefix
    summary.columns = [*group_cols, f"{label}_mean", f"{label}_std", "count"]
    return summary


def plot_method_lines(
    ax,
    summary: pd.DataFrame,
    x_col: str,
    y_col: str,
    methods: list[str] | None = None,
    y_std_col: str | None = None,
    show_legend: bool = True,
    legend_kwargs: dict | None = None,
) -> None:
    for method in ordered_methods(methods or summary["method"].unique().tolist()):
        method_data = summary[summary["method"] == method]
        if method_data.empty:
            continue

        style = style_for_method(method)
        x_values = np.asarray(method_data[x_col], dtype=float)
        y_values = np.asarray(method_data[y_col], dtype=float)
        ax.plot(
            x_values,
            y_values,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markeredgewidth=0,
            markersize=8,
            linewidth=2,
            alpha=1,
        )
        if y_std_col is not None:
            y_std = np.nan_to_num(np.asarray(method_data[y_std_col], dtype=float), nan=0.0)
            ax.fill_between(x_values, y_values - y_std, y_values + y_std, color=style["color"], alpha=0.2)

    if show_legend:
        ax.legend(numpoints=1,**(legend_kwargs or {"loc": "best", "fontsize": 10, "framealpha": 0.95}))


def configure_paper_style(plt) -> None:
    plt.style.use("classic")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "text.color": "#2F2840",
            "axes.facecolor": "white",
            "axes.edgecolor": "#8E84A8",
            "axes.labelcolor": "#2F2840",
            "axes.titlecolor": "#2F2840",
            "xtick.color": "#4E4464",
            "ytick.color": "#4E4464",
            "axes.unicode_minus": False,
        }
    )
    if shutil.which("latex") is not None:
        plt.rcParams["text.usetex"] = True


def configure_runtime_style(plt) -> None:
    configure_paper_style(plt)
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 10 / 72,
            "figure.constrained_layout.w_pad": 4 / 72,
            "figure.constrained_layout.hspace": 0.08,
            "figure.constrained_layout.wspace": 0.02,
            "savefig.dpi": 400,
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 15,
        }
    )


def load_best_result_entries(results: str | Path):
    results_dir = project_path(results)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    best_result_files = sorted(results_dir.glob("*/*/*_best_results.json"))
    if not best_result_files:
        raise ValueError(f"No best_results.json files found in {results_dir}")

    return [
        (best_file.parent.parent.name, best_file.parent.name, json.loads(best_file.read_text()))
        for best_file in best_result_files
    ]
