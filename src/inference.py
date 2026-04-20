"""Load saved hybrid model and run inference on raw text (Day 11)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.nlp_utils_news import SPACY_MODEL_SMALL, SpacyPreprocessor

DEFAULT_ARTIFACT = Path(__file__).resolve().parent.parent / "models" / "news_hybrid.joblib"


def default_artifact_path() -> Path:
    return DEFAULT_ARTIFACT


def load_artifact(path: Path | str | None = None) -> dict[str, Any]:
    """Load joblib bundle: ``{"pipeline": Pipeline, "classes": ndarray[str]}``."""
    p = Path(path) if path is not None else default_artifact_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {p}\nTrain first:  uv run python train.py"
        )
    return joblib.load(p)


def raw_text_to_frame(raw: str) -> pd.DataFrame:
    """
    Build the two-column input the hybrid expects: raw `text` + Day-8-style `text_clean`.
    """
    clean = SpacyPreprocessor(model_name=SPACY_MODEL_SMALL)
    return pd.DataFrame({"text": [raw], "text_clean": [clean(raw)]})


def predict_one(bundle: dict[str, Any], raw_text: str) -> str:
    """Return class name for a single raw document."""
    X = raw_text_to_frame(raw_text)
    y_idx = bundle["pipeline"].predict(X)[0]
    return str(bundle["classes"][int(y_idx)])


def predict_with_details(
    bundle: dict[str, Any], raw_text: str
) -> tuple[str, np.ndarray, np.ndarray | None]:
    """
    Return (predicted_label, class_names, decision_function scores or None).

    LinearSVC multiclass exposes ``decision_function`` with one score per class
    (larger = more confident toward that class).
    """
    X = raw_text_to_frame(raw_text)
    pipe = bundle["pipeline"]
    classes = np.asarray(bundle["classes"])
    y_idx = int(pipe.predict(X)[0])
    label = str(classes[y_idx])

    scores: np.ndarray | None = None
    if hasattr(pipe, "decision_function"):
        raw = pipe.decision_function(X)
        scores = np.asarray(raw)
        if scores.ndim == 2:
            scores = scores[0]

    return label, classes, scores
