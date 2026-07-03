"""Machine-learning layer: build dataset, walk-forward training, inference.

Lightweight by design — heavy libraries (lightgbm, scikit-learn) are imported
lazily so the rest of the package imports without them installed.
"""
