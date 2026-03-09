"""Evaluate Graph-CH profile candidates on existing grid-search outputs.

This script re-scores existing (alpha, t) clustering predictions with new
Graph-CH filter profiles, without re-running spectral clustering.
"""
# pyright: reportGeneralTypeIssues=false

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from utils.file_manager import load_dataset
from utils.metrics.graph_CH import (
    build_filtered_diffusion_embedding,
    build_legacy_graph_ch_profiles,
    build_research_graph_ch_profiles,
    graph_calinski_harabasz_from_embedding,
    merge_profiles,
)


def _safe_corr(x, y) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size < 2 or y_arr.size < 2:
        return np.nan, np.nan
    x0 = float(np.ravel(x_arr)[0])
    y0 = float(np.ravel(y_arr)[0])
    if np.allclose(x_arr, x0) or np.allclose(y_arr, y0):
        return np.nan, np.nan
    p_stat = np.corrcoef(x_arr, y_arr)[0, 1]
    rx = rankdata(x_arr)
    ry = rankdata(y_arr)
    s_stat = np.corrcoef(rx, ry)[0, 1]
    return float(p_stat), float(s_stat)


def _dataset_dirs(results_dir: Path, gsc_method: str, dataset_contains: str) -> list[Path]:
    target = f"{gsc_method}_all_results.json"
    out = []
    seen = set()
    for p in sorted(results_dir.rglob(target)):
        ds_dir = p.parent.parent
        rel = ds_dir.relative_to(results_dir).as_posix()
        if dataset_contains and dataset_contains not in rel:
            continue
        if ds_dir in seen:
            continue
        seen.add(ds_dir)
        out.append(ds_dir)
    return out


def _deduplicate_alpha_t(entries: list[dict]) -> list[dict]:
    seen: set[tuple[float, float]] = set()
    rows: list[dict] = []
    for e in entries:
        measure = e.get("measure", [None, {}])
        kwargs = {}
        if isinstance(measure, list) and len(measure) >= 2 and isinstance(measure[1], dict):
            kwargs = measure[1]

        alpha = float(kwargs.get("alpha", np.nan))
        t = float(kwargs.get("t", np.nan))
        if not np.isfinite(alpha) or not np.isfinite(t):
            continue

        key = (alpha, t)
        if key in seen:
            continue
        seen.add(key)

        labels = np.asarray(e.get("predicted_labels", [[]])[0], dtype=int)
        ami = float(e.get("ami", {}).get("mean", np.nan))
        rows.append({"alpha": alpha, "t": t, "ami": ami, "labels": labels})

    return rows


def _select_profile_catalog(profile_set: str, profile_ids: set[str]) -> list[dict]:
    profile_set = profile_set.lower()
    if profile_set not in {"legacy", "research", "extended"}:
        raise ValueError("profile_set must be one of legacy/research/extended")

    legacy = build_legacy_graph_ch_profiles()
    research = build_research_graph_ch_profiles()

    if profile_set == "legacy":
        profiles = legacy
    elif profile_set == "research":
        profiles = research
    else:
        profiles = merge_profiles(legacy, research)

    if profile_ids:
        profiles = [p for p in profiles if p["profile_id"] in profile_ids]
        if not profiles:
            raise ValueError("No profile_ids matched requested --profiles")

    return profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Graph-CH profile candidates from saved results")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--profile-set", default="research", help="legacy | research | extended")
    parser.add_argument("--profiles", default="", help="comma-separated profile_id filter")
    parser.add_argument("--dataset-contains", default="digrac_lead_lag_")
    parser.add_argument("--gsc-method", default="GSC-N")
    parser.add_argument("--sc-method", default="SC-N")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "analysis_candidate_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    profile_ids = {s.strip() for s in args.profiles.split(",") if s.strip()}
    profiles = _select_profile_catalog(args.profile_set, profile_ids)

    ds_dirs = _dataset_dirs(results_dir, args.gsc_method, args.dataset_contains)
    if not ds_dirs:
        raise RuntimeError("No matching dataset directories found")

    records = []
    pooled_pairs: dict[str, list[tuple[float, float]]] = {str(p["profile_id"]): [] for p in profiles}
    for ds_dir in ds_dirs:
        ds_name = ds_dir.relative_to(results_dir).as_posix()
        A, _ = load_dataset(args.datasets_root, ds_name)

        gsc_entries = json.loads((ds_dir / args.gsc_method / f"{args.gsc_method}_all_results.json").read_text())
        sc_entries = json.loads((ds_dir / args.sc_method / f"{args.sc_method}_all_results.json").read_text())
        sc_ami = float(sc_entries[0]["ami"]["mean"])

        at_rows = _deduplicate_alpha_t(gsc_entries)
        if not at_rows:
            continue
        oracle_ami = float(max(r["ami"] for r in at_rows))

        for profile in profiles:
            profile_id = str(profile["profile_id"])
            graph_cfg = profile.get("graph_ch", {})
            coeffs = graph_cfg.get("filter_coeffs", {1: 1.0})
            weighted = bool(graph_cfg.get("weighted", False))
            epsilon = float(graph_cfg.get("epsilon", 1e-10))

            Z = build_filtered_diffusion_embedding(A, filter_coeffs=coeffs, weighted=weighted, epsilon=epsilon)

            rows = []
            for r in at_rows:
                gch = graph_calinski_harabasz_from_embedding(Z, r["labels"])
                rows.append((r["alpha"], r["t"], r["ami"], gch))
                pooled_pairs[profile_id].append((float(gch), float(r["ami"])))

            df = pd.DataFrame(rows, columns=["alpha", "t", "ami", "graph_ch"])
            p, s = _safe_corr(df["graph_ch"].to_numpy(dtype=float), df["ami"].to_numpy(dtype=float))

            idx_sel = int(np.nanargmax(df["graph_ch"].to_numpy(dtype=float)))
            selected = df.iloc[idx_sel]
            selected_ami = float(selected["ami"])

            records.append(
                {
                    "dataset": ds_name,
                    "profile_id": profile_id,
                    "profile_family": profile.get("profile_family", "unknown"),
                    "profile_scale": float(profile.get("profile_scale", np.nan)),
                    "selected_alpha": float(selected["alpha"]),
                    "selected_t": float(selected["t"]),
                    "dataset_grid_pearson": p,
                    "dataset_grid_spearman": s,
                    "gsc_ami_selected": selected_ami,
                    "gsc_ami_oracle": oracle_ami,
                    "sc_ami": sc_ami,
                    "selection_regret": oracle_ami - selected_ami,
                    "selected_minus_sc": selected_ami - sc_ami,
                }
            )

    detailed = pd.DataFrame(records)
    if detailed.empty:
        raise RuntimeError("No candidate evaluation rows were produced")

    summary = detailed.groupby(["profile_id", "profile_family", "profile_scale"], as_index=False).agg(
        mean_dataset_pearson=("dataset_grid_pearson", "mean"),
        mean_dataset_spearman=("dataset_grid_spearman", "mean"),
        mean_gsc_selected_ami=("gsc_ami_selected", "mean"),
        mean_gsc_oracle_ami=("gsc_ami_oracle", "mean"),
        mean_sc_ami=("sc_ami", "mean"),
        mean_selection_regret=("selection_regret", "mean"),
        mean_selected_minus_sc=("selected_minus_sc", "mean"),
        n_datasets=("dataset", "nunique"),
    )
    sort_idx = np.lexsort(
        (
            -summary["mean_dataset_spearman"].to_numpy(dtype=float),
            -summary["mean_selected_minus_sc"].to_numpy(dtype=float),
            summary["mean_selection_regret"].to_numpy(dtype=float),
        )
    )
    summary = summary.iloc[sort_idx].reset_index(drop=True)

    # pooled (overall) correlations over all datasets and (alpha, t)
    pooled_rows = []
    for profile_id in summary["profile_id"]:
        pairs = pooled_pairs.get(str(profile_id), [])
        if pairs:
            x = [p[0] for p in pairs]
            y = [p[1] for p in pairs]
            p, s = _safe_corr(x, y)
        else:
            p, s = (np.nan, np.nan)
        pooled_rows.append({"profile_id": profile_id, "overall_pearson": p, "overall_spearman": s})
    pooled = pd.DataFrame(pooled_rows)
    summary = summary.merge(pooled, on="profile_id", how="left")

    detailed.to_csv(out_dir / "tables" / "dataset_profile_metrics.csv", index=False)
    summary.to_csv(out_dir / "tables" / "profile_summary.csv", index=False)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": results_dir.as_posix(),
        "profile_set": args.profile_set,
        "profile_count": int(len(profiles)),
        "dataset_count": int(pd.Series(detailed["dataset"]).nunique()),
        "best_profile": str(summary.iloc[0]["profile_id"]),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    top_cols = [
        "profile_id",
        "profile_family",
        "mean_selection_regret",
        "mean_selected_minus_sc",
        "mean_dataset_spearman",
    ]
    print(summary[top_cols].head(12).to_string(index=False))
    print(f"\nSaved candidate profile evaluation to: {out_dir}")


if __name__ == "__main__":
    main()
