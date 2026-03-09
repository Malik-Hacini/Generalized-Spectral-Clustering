"""Generate per-dataset alpha/t heatmaps for lead-lag GSC results.

For each lead-lag dataset, this script writes one PDF with two panels:
1) AMI over (alpha, t)
2) Graph-CH over (alpha, t)

By default, it auto-selects the most detailed lead-lag grid-search directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class GridSpecSummary:
    results_dir: Path
    dataset_sample: str
    entries: int
    alpha_count: int
    t_count: int
    profile_count: int

    @property
    def alpha_t_points(self) -> int:
        return self.alpha_count * self.t_count

    @property
    def effective_points(self) -> int:
        return self.alpha_count * self.t_count * max(1, self.profile_count)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _parse_alpha_t(entry: dict) -> tuple[float, float]:
    measure = entry.get("measure", [None, {}])
    kwargs = {}
    if isinstance(measure, list) and len(measure) >= 2 and isinstance(measure[1], dict):
        kwargs = measure[1]
    alpha = float(kwargs.get("alpha", np.nan))
    t = float(kwargs.get("t", np.nan))
    return alpha, t


def _parse_profile_id(entry: dict) -> str | None:
    metric_params = entry.get("metric_params")
    if isinstance(metric_params, dict):
        profile_id = metric_params.get("profile_id")
        if profile_id is not None:
            return str(profile_id)
    return None


def _extract_mean(entry: dict, metric_name: str) -> float:
    payload = entry.get(metric_name, {})
    if not isinstance(payload, dict):
        return np.nan
    value = payload.get("mean", np.nan)
    return float(value)


def _candidate_lead_lag_results_dirs(results_root: Path) -> list[Path]:
    candidates = []
    for p in sorted(results_root.glob("benchmark_lead_lag*grid_search")):
        if p.is_dir():
            candidates.append(p)
    return candidates


def _sample_gsc_file(results_dir: Path, gsc_method: str, dataset_contains: str) -> Path | None:
    target = f"{gsc_method}_all_results.json"
    for p in sorted(results_dir.rglob(target)):
        if dataset_contains in p.as_posix():
            return p
    return None


def _summarize_results_dir(results_dir: Path, gsc_method: str, dataset_contains: str) -> GridSpecSummary | None:
    sample = _sample_gsc_file(results_dir, gsc_method, dataset_contains)
    if sample is None:
        return None

    rows = json.loads(sample.read_text())
    alphas = []
    ts = []
    profiles = []
    for r in rows:
        alpha, t = _parse_alpha_t(r)
        if np.isfinite(alpha):
            alphas.append(alpha)
        if np.isfinite(t):
            ts.append(t)
        profile_id = _parse_profile_id(r)
        if profile_id is not None:
            profiles.append(profile_id)

    return GridSpecSummary(
        results_dir=results_dir,
        dataset_sample=sample.parent.parent.name,
        entries=len(rows),
        alpha_count=len(set(alphas)),
        t_count=len(set(ts)),
        profile_count=len(set(profiles)),
    )


def _pick_most_detailed_dir(results_root: Path, gsc_method: str, dataset_contains: str) -> tuple[Path, list[GridSpecSummary]]:
    summaries = []
    for d in _candidate_lead_lag_results_dirs(results_root):
        summary = _summarize_results_dir(d, gsc_method=gsc_method, dataset_contains=dataset_contains)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise RuntimeError(f"No lead-lag result directories found under {results_root}")

    summaries = sorted(
        summaries,
        key=lambda s: (s.effective_points, s.alpha_t_points, s.entries),
        reverse=True,
    )
    return summaries[0].results_dir, summaries


def _choose_profile_id(results_dir: Path, explicit_profile: str | None) -> str | None:
    if explicit_profile:
        return explicit_profile

    profile_summary_path = results_dir / "analysis_graphch_profiles" / "tables" / "profile_summary.csv"
    if profile_summary_path.exists():
        df = pd.read_csv(profile_summary_path)
        if not df.empty and "profile_id" in df.columns:
            return str(df.iloc[0]["profile_id"])

    sample = _sample_gsc_file(results_dir, gsc_method="GSC-N", dataset_contains="digrac_lead_lag_")
    if sample is None:
        return None

    rows = json.loads(sample.read_text())
    profiles = [p for p in (_parse_profile_id(r) for r in rows) if p is not None]
    if profiles:
        return sorted(set(profiles))[0]
    return None


def _dataset_dirs(results_dir: Path, gsc_method: str, dataset_contains: str) -> list[Path]:
    out = []
    target = f"{gsc_method}_all_results.json"
    seen = set()
    for p in sorted(results_dir.rglob(target)):
        if dataset_contains not in p.as_posix():
            continue
        d = p.parent.parent
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


def _build_dataset_frame(gsc_entries: list[dict], profile_id: str | None) -> pd.DataFrame:
    rows = []
    for e in gsc_entries:
        this_profile = _parse_profile_id(e)
        if profile_id is not None and this_profile != profile_id:
            continue
        alpha, t = _parse_alpha_t(e)
        if not np.isfinite(alpha) or not np.isfinite(t):
            continue
        rows.append(
            {
                "alpha": float(alpha),
                "t": float(t),
                "ami": _extract_mean(e, "ami"),
                "graph_ch": _extract_mean(e, "graph_ch"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows found for selected profile")
    return df


def _matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    mat = df.pivot_table(index="alpha", columns="t", values=value_col, aggfunc="mean")
    mat = mat.sort_index(axis=0).sort_index(axis=1)
    return mat


def _marker_index(mat: pd.DataFrame, mode: str = "max") -> tuple[int, int]:
    values = mat.to_numpy(dtype=float)
    if mode == "max":
        idx = np.nanargmax(values)
    else:
        idx = np.nanargmin(values)
    i, j = np.unravel_index(idx, values.shape)
    return int(i), int(j)


def _plot_heatmap(ax, mat: pd.DataFrame, title: str, cmap: str, marker_mode: str = "max") -> tuple[float, float, float]:
    y = mat.index.to_numpy(dtype=float)
    x = mat.columns.to_numpy(dtype=float)
    values = mat.to_numpy(dtype=float)

    im = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("alpha")

    x_ticks = np.arange(len(x))
    y_ticks = np.arange(len(y))
    ax.set_xticks(x_ticks[:: max(1, len(x) // 8)])
    ax.set_yticks(y_ticks[:: max(1, len(y) // 8)])
    ax.set_xticklabels([f"{xv:g}" for xv in x[:: max(1, len(x) // 8)]])
    ax.set_yticklabels([f"{yv:g}" for yv in y[:: max(1, len(y) // 8)]])

    i, j = _marker_index(mat, mode=marker_mode)
    ax.scatter(j, i, marker="*", s=130, c="white", edgecolors="black", linewidths=1.0, zorder=4)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)

    return float(y[i]), float(x[j]), float(values[i, j])


def _render_dataset_pdf(
    dataset_name: str,
    mat_ami: pd.DataFrame,
    mat_graph_ch: pd.DataFrame,
    sc_ami: float,
    out_path: Path,
    profile_id: str | None,
) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

    alpha_oracle, t_oracle, ami_oracle = _plot_heatmap(
        axes[0], mat_ami, title="AMI over (alpha, t)", cmap="viridis", marker_mode="max"
    )
    alpha_sel, t_sel, graph_ch_best = _plot_heatmap(
        axes[1], mat_graph_ch, title="Graph-CH over (alpha, t)", cmap="magma", marker_mode="max"
    )

    selected_ami = float(mat_ami.loc[alpha_sel, t_sel])
    regret = ami_oracle - selected_ami
    gain_vs_sc = selected_ami - sc_ami

    profile_txt = profile_id if profile_id is not None else "none"
    fig.suptitle(
        (
            f"{dataset_name} | profile={profile_txt} | "
            f"SC={sc_ami:.4f}, selected={selected_ami:.4f}, oracle={ami_oracle:.4f}, "
            f"regret={regret:.4f}, sel-SC={gain_vs_sc:.4f}"
        ),
        fontsize=10,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "dataset": dataset_name,
        "profile_id": profile_id,
        "sc_ami": sc_ami,
        "selected_ami": selected_ami,
        "oracle_ami": ami_oracle,
        "selection_regret": regret,
        "selected_minus_sc": gain_vs_sc,
        "selected_alpha": alpha_sel,
        "selected_t": t_sel,
        "oracle_alpha": alpha_oracle,
        "oracle_t": t_oracle,
        "selected_graph_ch": graph_ch_best,
        "pdf_path": out_path.as_posix(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-dataset lead-lag alpha/t AMI and Graph-CH heatmaps")
    parser.add_argument("--results-root", default="results", help="Root directory containing benchmark results")
    parser.add_argument("--results-dir", default=None, help="Explicit lead-lag results directory")
    parser.add_argument("--gsc-method", default="GSC-N")
    parser.add_argument("--sc-method", default="SC-N")
    parser.add_argument("--profile-id", default=None, help="Graph-CH profile id (default: best from profile_summary.csv)")
    parser.add_argument("--dataset-contains", default="digrac_lead_lag_", help="Substring used to filter datasets")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <results-dir>/analysis_alpha_t_heatmaps/<profile_id>)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _style()

    results_root = Path(args.results_root)

    if args.results_dir:
        results_dir = Path(args.results_dir)
        summaries = []
    else:
        results_dir, summaries = _pick_most_detailed_dir(
            results_root=results_root,
            gsc_method=args.gsc_method,
            dataset_contains=args.dataset_contains,
        )

    profile_id = _choose_profile_id(results_dir=results_dir, explicit_profile=args.profile_id)
    profile_slug = profile_id if profile_id is not None else "no_profile"

    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "analysis_alpha_t_heatmaps" / profile_slug
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    manifest_dir = out_dir / "manifests"
    for d in [table_dir, fig_dir, manifest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_dir in _dataset_dirs(results_dir, gsc_method=args.gsc_method, dataset_contains=args.dataset_contains):
        dataset_name = dataset_dir.relative_to(results_dir).as_posix()
        gsc_path = dataset_dir / args.gsc_method / f"{args.gsc_method}_all_results.json"
        sc_path = dataset_dir / args.sc_method / f"{args.sc_method}_all_results.json"

        gsc_entries = json.loads(gsc_path.read_text())
        sc_entries = json.loads(sc_path.read_text())
        sc_ami = _extract_mean(sc_entries[0], "ami") if sc_entries else np.nan

        df = _build_dataset_frame(gsc_entries=gsc_entries, profile_id=profile_id)
        mat_ami = _matrix(df, value_col="ami")
        mat_graph_ch = _matrix(df, value_col="graph_ch")

        out_pdf = fig_dir / f"{dataset_dir.name}__ami_graphch_alpha_t.pdf"
        row = _render_dataset_pdf(
            dataset_name=dataset_name,
            mat_ami=mat_ami,
            mat_graph_ch=mat_graph_ch,
            sc_ami=sc_ami,
            out_path=out_pdf,
            profile_id=profile_id,
        )
        row["alpha_count"] = int(mat_ami.shape[0])
        row["t_count"] = int(mat_ami.shape[1])
        row["grid_points"] = int(mat_ami.shape[0] * mat_ami.shape[1])
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    summary_df.to_csv(table_dir / "dataset_heatmap_summary.csv", index=False)

    detail_rows = []
    if summaries:
        for s in summaries:
            detail_rows.append(
                {
                    "results_dir": s.results_dir.as_posix(),
                    "dataset_sample": s.dataset_sample,
                    "entries_per_dataset": s.entries,
                    "alpha_count": s.alpha_count,
                    "t_count": s.t_count,
                    "profile_count": s.profile_count,
                    "alpha_t_points": s.alpha_t_points,
                    "effective_points": s.effective_points,
                }
            )
    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df.to_csv(table_dir / "candidate_results_grid_sizes.csv", index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": results_dir.as_posix(),
        "profile_id": profile_id,
        "n_datasets": int(len(summary_df)),
        "grid": {
            "alpha_count": int(summary_df["alpha_count"].iloc[0]) if not summary_df.empty else None,
            "t_count": int(summary_df["t_count"].iloc[0]) if not summary_df.empty else None,
            "grid_points": int(summary_df["grid_points"].iloc[0]) if not summary_df.empty else None,
        },
    }
    (manifest_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Selected results dir: {results_dir}")
    print(f"Selected profile: {profile_id}")
    if not summary_df.empty:
        print(
            "Grid size per dataset: "
            f"alpha={int(summary_df['alpha_count'].iloc[0])}, "
            f"t={int(summary_df['t_count'].iloc[0])}, "
            f"alpha*t={int(summary_df['grid_points'].iloc[0])}"
        )
    print(f"Saved per-dataset heatmaps to: {fig_dir}")


if __name__ == "__main__":
    main()
