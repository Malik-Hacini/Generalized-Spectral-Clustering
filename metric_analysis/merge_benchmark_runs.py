"""Merge separate DSBM benchmark runs into a single analysis-ready tree.

Use case:
- Run A contains AMI + Graph-CH profile grid results.
- Run B contains AMI + modularity + map equation results.

This script builds a merged run where each grid entry contains all metrics:
``ami, graph_ch, modularity, map_equation``.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_RE = re.compile(r"^dsbm_gamma(?P<gamma>\d+(?:\.\d+)?)_seed(?P<seed>\d+)$")
METRICS = ("ami", "graph_ch", "modularity", "map_equation")
MINIMIZE_METRICS = {"map_equation"}


def _dataset_names(root: Path) -> list[str]:
    names: list[tuple[float, int, str]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        m = DATASET_RE.match(entry.name)
        if m is None:
            continue
        names.append((float(m.group("gamma")), int(m.group("seed")), entry.name))
    names.sort(key=lambda x: (x[0], x[1]))
    return [name for _, _, name in names]


def _json_load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_dump(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _measure_key(entry: dict) -> str:
    measure = entry.get("measure", [None, {}])
    if isinstance(measure, list) and len(measure) >= 2 and isinstance(measure[1], dict):
        kwargs = measure[1]
        return json.dumps(kwargs, sort_keys=True, default=str)
    return "{}"


def _extract_mean(entry: dict, metric: str) -> float:
    payload = entry[metric]
    if isinstance(payload, dict) and "mean" in payload:
        return float(payload["mean"])
    return float(payload)


def _compute_best_results(all_results: list[dict]) -> dict:
    best_results = {}
    for metric in METRICS:
        values = np.asarray([_extract_mean(entry, metric) for entry in all_results], dtype=float)
        if metric in MINIMIZE_METRICS:
            idx = int(np.argmin(values))
        else:
            idx = int(np.argmax(values))
        best_results[metric] = copy.deepcopy(all_results[idx])
    return best_results


def _compute_summary_row(method_name: str, all_results: list[dict]) -> dict:
    row = {"method": method_name}
    for metric in METRICS:
        values = np.asarray([_extract_mean(entry, metric) for entry in all_results], dtype=float)
        row[f"{metric}_mean_across_grid"] = round(float(np.mean(values)), 3)
        row[f"{metric}_std_across_grid"] = round(float(np.std(values)), 3)

        if metric in MINIMIZE_METRICS:
            idx = int(np.argmin(values))
        else:
            idx = int(np.argmax(values))
        best_metric_payload = all_results[idx][metric]
        row[f"{metric}_best_mean"] = round(float(best_metric_payload["mean"]), 3)
        row[f"{metric}_best_std"] = round(float(best_metric_payload["std"]), 3)
    return row


def _merge_method_results(
    graphch_all: list[dict],
    modmap_all: list[dict],
    method_name: str,
) -> list[dict]:
    if method_name == "GSC-N":
        lookup = {}
        for entry in modmap_all:
            key = _measure_key(entry)
            lookup[key] = {
                "modularity": copy.deepcopy(entry["modularity"]),
                "map_equation": copy.deepcopy(entry["map_equation"]),
            }

        merged = []
        for entry in graphch_all:
            key = _measure_key(entry)
            if key not in lookup:
                raise RuntimeError(f"Missing modularity/map_equation match for GSC measure key: {key}")
            out = copy.deepcopy(entry)
            out["modularity"] = copy.deepcopy(lookup[key]["modularity"])
            out["map_equation"] = copy.deepcopy(lookup[key]["map_equation"])
            merged.append(out)
        return merged

    if method_name == "SC-N":
        if not modmap_all:
            raise RuntimeError("SC-N mod/map run has no entries")
        mod_payload = copy.deepcopy(modmap_all[0]["modularity"])
        map_payload = copy.deepcopy(modmap_all[0]["map_equation"])

        merged = []
        for entry in graphch_all:
            out = copy.deepcopy(entry)
            out["modularity"] = copy.deepcopy(mod_payload)
            out["map_equation"] = copy.deepcopy(map_payload)
            merged.append(out)
        return merged

    raise ValueError(f"Unsupported method: {method_name}")


def merge_runs(
    graphch_root: Path,
    modmap_root: Path,
    output_root: Path,
    overwrite: bool,
) -> None:
    if not graphch_root.exists():
        raise FileNotFoundError(f"Graph-CH run not found: {graphch_root}")
    if not modmap_root.exists():
        raise FileNotFoundError(f"Modularity/map-equation run not found: {modmap_root}")

    ds_graphch = _dataset_names(graphch_root)
    ds_modmap = _dataset_names(modmap_root)
    if ds_graphch != ds_modmap:
        missing_in_modmap = sorted(set(ds_graphch) - set(ds_modmap))
        missing_in_graphch = sorted(set(ds_modmap) - set(ds_graphch))
        raise RuntimeError(
            "Dataset mismatch between runs. "
            f"Missing in modmap: {missing_in_modmap[:5]}; "
            f"Missing in graphch: {missing_in_graphch[:5]}"
        )

    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")

    shutil.copytree(graphch_root, output_root)

    for dataset in ds_graphch:
        graph_dataset = graphch_root / dataset
        modmap_dataset = modmap_root / dataset
        out_dataset = output_root / dataset

        summary_rows = []
        for method_name in ("SC-N", "GSC-N"):
            graph_all_path = graph_dataset / method_name / f"{method_name}_all_results.json"
            modmap_all_path = modmap_dataset / method_name / f"{method_name}_all_results.json"
            out_all_path = out_dataset / method_name / f"{method_name}_all_results.json"
            out_best_path = out_dataset / method_name / f"{method_name}_best_results.json"

            graphch_all = _json_load(graph_all_path)
            modmap_all = _json_load(modmap_all_path)

            merged_all = _merge_method_results(graphch_all, modmap_all, method_name)
            _json_dump(out_all_path, merged_all)

            best_results = _compute_best_results(merged_all)
            _json_dump(out_best_path, best_results)

            summary_rows.append(_compute_summary_row(method_name, merged_all))

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dataset / f"{dataset}_summary.csv", index=False)

    source_param_candidates = sorted(graphch_root.glob("*_params.json"))
    if source_param_candidates:
        merged_param_path = output_root / "benchmark_dsbm_all_metrics_profiles_params.json"
        shutil.copy2(source_param_candidates[0], merged_param_path)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_graphch_root": str(graphch_root),
        "source_modmap_root": str(modmap_root),
        "output_root": str(output_root),
        "dataset_count": len(ds_graphch),
        "metrics": list(METRICS),
        "minimize_metrics": sorted(MINIMIZE_METRICS),
    }
    _json_dump(output_root / "merge_manifest.json", manifest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge split DSBM benchmark runs into one all-metrics tree")
    parser.add_argument(
        "--graphch-root",
        default="results/benchmark_dsbm_graphch_profiles_grid_search",
        help="Source run containing graph_ch profile sweep",
    )
    parser.add_argument(
        "--modmap-root",
        default="results/benchmark_dsbm_grid_search",
        help="Source run containing modularity and map_equation",
    )
    parser.add_argument(
        "--output-root",
        default="results/benchmark_dsbm_all_metrics_profiles_grid_search",
        help="Target merged benchmark directory",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    merge_runs(
        graphch_root=Path(args.graphch_root),
        modmap_root=Path(args.modmap_root),
        output_root=Path(args.output_root),
        overwrite=args.overwrite,
    )
    print(f"Merged benchmark written to: {args.output_root}")


if __name__ == "__main__":
    main()
