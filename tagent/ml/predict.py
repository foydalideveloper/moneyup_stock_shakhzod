"""Model inference wrapper.

Loads a saved model bundle and turns a recent OHLCV window into a probability
that the current bar is a good long entry (per the triple-barrier label).
"""

from __future__ import annotations

import pickle
from typing import Optional

import pandas as pd

from tagent.config import MODEL_DIR
from tagent.features import make_features


class Predictor:
    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.features = list(bundle["features"])
        self.metrics = bundle.get("metrics", {})

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Predictor":
        path = str(path or (MODEL_DIR / "model.pkl"))
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
        return cls(bundle)

    def predict_proba_latest(self, ohlcv: pd.DataFrame) -> float:
        """Probability (0..1) that the most recent bar is a good entry."""
        feats = make_features(ohlcv, dropna=True)
        if feats.empty:
            raise ValueError("Not enough data to compute features.")
        row = feats.iloc[[-1]]
        # Align to the exact feature set the model was trained on.
        row = row.reindex(columns=self.features)
        proba = self.model.predict_proba(row)[:, 1]
        return float(proba[0])
