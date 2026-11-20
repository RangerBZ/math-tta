"""Shared progress-bar dependency loading."""

from __future__ import annotations

from typing import Any


def load_tqdm() -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        raise RuntimeError(
            "Progress bars require tqdm. Install generation dependencies with "
            '`python -m pip install -e ".[model]"` or training dependencies '
            'with `python -m pip install -e ".[train]"`.'
        ) from error
    return tqdm
