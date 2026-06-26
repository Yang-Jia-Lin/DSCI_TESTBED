"""Small shared serialization and file-opening helpers."""

import json
import os
from pathlib import Path

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return super().default(value)


def open_file(path: str | Path) -> None:
    os.startfile(Path(path))
