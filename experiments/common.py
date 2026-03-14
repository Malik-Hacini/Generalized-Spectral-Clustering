from __future__ import annotations

from pathlib import Path

import numpy as np

from competitors.disim import avg_deg_taus
from competitors.measures import *  # noqa: F401,F403
from competitors.neighbors import log_neighbors
from utils.config import ExperimentConfig
from utils.experiments_utils import experiment

ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> str:
    path = Path(path)
    return str(path if path.is_absolute() else ROOT / path)
