"""Models.ModelNet compatibility package.

Maps `Models.ModelNet.*` imports to the actual implementation under
`Models/Models/*`.

This file contains no business logic.
"""

from __future__ import annotations

from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_models_dir = _pkg_dir.parent
_models_models_dir = _models_dir / "Models"

__path__ = [
    str(_pkg_dir),
    str(_models_models_dir),
]
