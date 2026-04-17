"""
Hybrid TF-IDF + spaCy linguistic pipeline (Day 10 / 11).

Single source of truth for `train.py`, `predict.py`, and notebooks.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.nlp_utils_news import SPACY_MODEL_SMALL, SpacyFeatureExtractor

TFIDF_KW = dict(
    ngram_range=(1, 2),
    min_df=2,
    max_features=30_000,
    sublinear_tf=True,
)


def build_hybrid_pipeline() -> Pipeline:
    """Same ColumnTransformer + LinearSVC stack as `day10_feature_engineering.ipynb`."""
    ling_pipe = Pipeline(
        [
            ("extract", SpacyFeatureExtractor(model_name=SPACY_MODEL_SMALL)),
            ("scale", StandardScaler()),
        ]
    )
    hybrid_features = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(**TFIDF_KW), "text_clean"),
            ("ling", ling_pipe, "text"),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        # Fit/transform columns sequentially — parallel branches duplicate spaCy + CPU load
        n_jobs=1,
    )
    return Pipeline(
        [
            ("features", hybrid_features),
            # High-dim TF-IDF + dense branch — liblinear often needs > default 1000 iters
            ("clf", LinearSVC(C=0.5, random_state=42, max_iter=20_000)),
        ]
    )
