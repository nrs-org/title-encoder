"""Project paths shared by the standalone training commands."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get("TITLE_ENCODER_DATA_ROOT", PROJECT_ROOT / "data" / "title-encoder")
).expanduser().resolve()


def build_dir(name: str) -> Path:
    return DATA_ROOT / name


def model_dir(name: str) -> Path:
    return DATA_ROOT / "models" / name
