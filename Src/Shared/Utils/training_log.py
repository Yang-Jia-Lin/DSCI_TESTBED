"""Training-only persistence helpers shared by offline entrypoints."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import torch


def save_model_weights(
    model: torch.nn.Module, bundle_id: str, weights_dir: Path
) -> Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    model_path = (
        weights_dir / f"{bundle_id}_{datetime.now().strftime('%m%d_%H%M')}.pth"
    )
    torch.save(model.state_dict(), model_path)
    return model_path


def save_train_log(
    log_data: Dict[str, Union[List[float], List[int]]], bundle_id: str, log_dir: Path
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"training_log_{bundle_id}_{datetime.now().strftime('%m%d_%H%M')}.csv"
    csv_path = log_dir / filename
    pd.DataFrame(log_data).to_csv(csv_path, index=False)
    return csv_path
