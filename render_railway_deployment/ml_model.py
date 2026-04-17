from typing import Any

import joblib


def load_model(path: str) -> Any:
    """Load trained ML pipeline from disk."""
    return joblib.load(path)
