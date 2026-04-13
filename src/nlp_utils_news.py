"""
nlp_utils_news.py
-----------------
Reusable spaCy + sklearn components for the 6-class news categorisation project.

Components
----------
SpacyPreprocessor     - callable that lemmatises + removes stopwords/punct
SpacyFeatureExtractor - sklearn-compatible transformer (fit/transform)
                        that extracts dense linguistic features:
                        • NER type counts  (ORG, GPE, PERSON, DATE, MONEY)
                        • POS ratio counts (NOUN, VERB, ADJ, ADV)
                        • Surface stats   (avg token len, n_sentences)

Usage
-----
    from src.nlp_utils_news import SpacyPreprocessor, SpacyFeatureExtractor

    # As a standalone text cleaner
    clean = SpacyPreprocessor()
    print(clean("Scientists at NASA discover new climate patterns."))

    # As a sklearn transformer inside Pipeline / FeatureUnion
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("tfidf",      TfidfVectorizer(ngram_range=(1, 2))),
            ("linguistic", SpacyFeatureExtractor()),
        ])),
        ("clf", LinearSVC(C=0.5)),
    ])
"""

from __future__ import annotations

import re
import warnings
from typing import Iterable

import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Module-level lazy loader — only loads the model once per Python process
# ---------------------------------------------------------------------------
_NLP_CACHE: dict[str, spacy.language.Language] = {}

SPACY_MODEL_SMALL  = "en_core_web_sm"   # Day 8 / 9  — no word vectors, fast
SPACY_MODEL_MEDIUM = "en_core_web_md"   # Day 10     — 300-d GloVe vectors

# Entity types used for linguistic features
NER_TYPES = ["ORG", "GPE", "PERSON", "DATE", "MONEY", "NORP", "FAC"]
# POS tags used for linguistic features
POS_TYPES = ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]


def _load_nlp(model_name: str = SPACY_MODEL_SMALL) -> spacy.language.Language:
    """Load + cache a spaCy model (avoids reloading in long-running sessions)."""
    if model_name not in _NLP_CACHE:
        try:
            _NLP_CACHE[model_name] = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"spaCy model '{model_name}' not found.\n"
                f"Install it with:  python -m spacy download {model_name}"
            )
    return _NLP_CACHE[model_name]


# ---------------------------------------------------------------------------
# 1. Simple text cleaner (used by TfidfVectorizer's `preprocessor` arg)
# ---------------------------------------------------------------------------

class SpacyPreprocessor:
    """
    Callable text cleaner that uses spaCy for lemmatisation.

    Steps
    -----
    1. Lower-case
    2. Strip HTML entities and URLs
    3. spaCy tokenise
    4. Drop stop-words, punctuation, spaces, single characters
    5. Lemmatise each surviving token

    Parameters
    ----------
    model_name : str
        spaCy model to load (default: en_core_web_sm).
    extra_stopwords : set[str] | None
        Additional domain-specific stop-words to remove.

    Examples
    --------
    >>> clean = SpacyPreprocessor()
    >>> clean("NASA scientists discovered new exoplanets orbiting distant stars.")
    'nasa scientist discover new exoplanet orbit distant star'
    """

    def __init__(
        self,
        model_name: str = SPACY_MODEL_SMALL,
        extra_stopwords: set[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.extra_stopwords = extra_stopwords or set()

    def __call__(self, text: str) -> str:
        return self._preprocess(text)

    def _preprocess(self, text: str) -> str:
        # ---- raw cleaning ----
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
        text = re.sub(r"&[a-z]+;", " ", text)                   # HTML entities
        text = re.sub(r"[^a-z\s]", " ", text)                   # keep only letters
        text = re.sub(r"\s+", " ", text).strip()

        # ---- spaCy pipeline ----
        nlp = _load_nlp(self.model_name)
        doc = nlp(text)

        tokens = [
            t.lemma_
            for t in doc
            if (
                not t.is_stop
                and not t.is_punct
                and not t.is_space
                and len(t.lemma_) > 2
                and t.lemma_ not in self.extra_stopwords
                and t.is_alpha
            )
        ]
        return " ".join(tokens)

    # Make it picklable (sklearn clones call __init__)
    def get_params(self, deep: bool = True) -> dict:
        return {"model_name": self.model_name, "extra_stopwords": self.extra_stopwords}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
# 2. Dense linguistic feature extractor (sklearn transformer)
# ---------------------------------------------------------------------------

class SpacyFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that converts raw text into a dense matrix
    of hand-crafted linguistic features using spaCy.

    Feature vector per document (len = len(NER_TYPES) + len(POS_TYPES) + 2):
    ┌──────────────────────────────────┐
    │ NER counts (normalised)          │  7 features
    │ POS ratio counts (normalised)    │  5 features
    │ avg_token_len                    │  1 feature
    │ n_sentences (log-scaled)         │  1 feature
    └──────────────────────────────────┘
                               Total = 14 features

    These are combined with TF-IDF via sklearn's FeatureUnion on Day 10.

    Parameters
    ----------
    model_name : str
        spaCy model to load (default: en_core_web_sm).

    Examples
    --------
    >>> extractor = SpacyFeatureExtractor()
    >>> X = extractor.fit_transform(["Obama spoke at the White House today."])
    >>> X.shape
    (1, 14)
    """

    def __init__(self, model_name: str = SPACY_MODEL_SMALL) -> None:
        self.model_name = model_name

    def fit(self, X: Iterable[str], y=None):
        """No fitting needed — features are rule-based."""
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        nlp = _load_nlp(self.model_name)
        rows = []
        for text in X:
            doc = nlp(str(text)[:50_000])  # cap at 50K chars for speed
            rows.append(self._extract(doc))
        return np.array(rows, dtype=np.float32)

    def _extract(self, doc: spacy.tokens.Doc) -> list[float]:
        n_tokens = max(len(doc), 1)

        # NER counts (normalised by doc length)
        ner_counts = {ner: 0 for ner in NER_TYPES}
        for ent in doc.ents:
            if ent.label_ in ner_counts:
                ner_counts[ent.label_] += 1
        ner_features = [ner_counts[n] / n_tokens for n in NER_TYPES]

        # POS ratios
        pos_counts = {pos: 0 for pos in POS_TYPES}
        for token in doc:
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1
        pos_features = [pos_counts[p] / n_tokens for p in POS_TYPES]

        # Surface stats
        alpha_tokens = [t for t in doc if t.is_alpha]
        avg_token_len = (
            np.mean([len(t.text) for t in alpha_tokens]) if alpha_tokens else 0.0
        )
        n_sents = max(len(list(doc.sents)), 1)
        log_sents = float(np.log1p(n_sents))

        return ner_features + pos_features + [float(avg_token_len), log_sents]

    def get_feature_names_out(self) -> list[str]:
        """Return human-readable feature names (works with sklearn's set_output API)."""
        ner_names = [f"ner_{n.lower()}" for n in NER_TYPES]
        pos_names = [f"pos_{p.lower()}" for p in POS_TYPES]
        return ner_names + pos_names + ["avg_token_len", "log_n_sents"]


# ---------------------------------------------------------------------------
# 3. Utility: batch-preprocess a pandas Series with progress bar
# ---------------------------------------------------------------------------

def preprocess_series(
    series,
    model_name: str = SPACY_MODEL_SMALL,
    batch_size: int = 512,
    n_process: int = 1,
) -> list[str]:
    """
    Efficiently preprocess a pandas Series of texts using spaCy's pipe().

    Parameters
    ----------
    series : pd.Series
        Raw text column.
    model_name : str
        spaCy model to use.
    batch_size : int
        Number of docs per batch (tune for RAM vs. speed).
    n_process : int
        Parallel workers (1 = single-threaded, safe on Windows).

    Returns
    -------
    list[str]
        Cleaned text strings in the same order as input.
    """
    try:
        from tqdm.auto import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    nlp = _load_nlp(model_name)
    preprocessor = SpacyPreprocessor(model_name=model_name)

    texts = series.fillna("").astype(str).tolist()
    results: list[str] = []

    # Disable components we don't need for speed (keep tagger, lemmatizer, ner)
    with nlp.select_pipes(disable=["parser"]):
        pipe = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
        if use_tqdm:
            pipe = tqdm(pipe, total=len(texts), desc="spaCy preprocessing")
        for doc in pipe:
            tokens = [
                t.lemma_.lower()
                for t in doc
                if not t.is_stop
                and not t.is_punct
                and not t.is_space
                and len(t.lemma_) > 2
                and t.is_alpha
            ]
            results.append(" ".join(tokens))

    return results


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing SpacyPreprocessor...")
    clean = SpacyPreprocessor()
    samples = [
        "NASA scientists discovered new exoplanets orbiting distant stars.",
        "The president signed a new healthcare bill at the White House.",
        "Manchester United won the Premier League after a thrilling season.",
        "Apple unveiled its latest iPhone with improved AI features.",
    ]
    for s in samples:
        print(f"  IN : {s}")
        print(f"  OUT: {clean(s)}\n")

    print("Testing SpacyFeatureExtractor...")
    extractor = SpacyFeatureExtractor()
    X = extractor.fit_transform(samples)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Feature names: {extractor.get_feature_names_out()}")
    print("  All tests passed!")
