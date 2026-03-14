from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np

from competitors import measures as _measures
from competitors.disim import avg_deg_taus
from competitors.measures import *  # noqa: F401,F403
from competitors.neighbors import log_neighbors
from utils.config import ExperimentConfig
from utils.experiments_utils import experiment

__all__ = [
    "np",
    "avg_deg_taus",
    "log_neighbors",
    "ExperimentConfig",
    "experiment",
    "project_path",
    *[name for name in dir(_measures) if not name.startswith("_")],
]


def project_path(path: str | Path) -> str:
    path = Path(path)
    return str(path if path.is_absolute() else ROOT / path)
