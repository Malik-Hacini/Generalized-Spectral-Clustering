from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOTS_ROOT = ROOT / "plots"


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def experiment_name(source: str | Path) -> str:
    source_path = project_path(source)
    return source_path.name if source_path.is_dir() else source_path.parent.name


def resolve_output_dir(output_dir: str | Path | None, kind: str, source: str | Path) -> Path:
    if output_dir is None:
        output_path = PLOTS_ROOT / kind / experiment_name(source)
    else:
        output_path = project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def resolve_output_file(
    output_dir: str | Path | None,
    output_name: str | None,
    kind: str,
    source: str | Path,
    default_name: str,
) -> Path:
    output_path = resolve_output_dir(output_dir, kind, source) / (output_name or default_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def validate_selection(available: list[str], selected: list[str] | None, label: str) -> list[str]:
    if selected is None:
        return available
    missing = [item for item in selected if item not in available]
    if missing:
        raise ValueError(f"Unknown {label}: {missing}. Available {label}: {available}")
    return selected


def load_best_result_entries(results: str | Path):
    results_dir = project_path(results)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    best_result_files = sorted(results_dir.glob("*/*/*_best_results.json"))
    if not best_result_files:
        raise ValueError(f"No best_results.json files found in {results_dir}")

    entries = []
    for best_file in best_result_files:
        entries.append(
            (
                best_file.parent.parent.name,
                best_file.parent.name,
                json.loads(best_file.read_text()),
            )
        )
    return entries
