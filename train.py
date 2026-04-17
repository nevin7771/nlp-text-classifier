#!/usr/bin/env python3
"""Train hybrid pipeline on parquet from Day 8 and save a joblib artifact (Day 11)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from src.pipeline_news import build_hybrid_pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_OUT = ROOT / "models" / "news_hybrid.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save 6-class news hybrid model.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output joblib path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--skip-test-metrics",
        action="store_true",
        help="Do not load test parquet or print test metrics",
    )
    args = parser.parse_args()

    train_pq = DATA_DIR / "news_train_clean.parquet"
    test_pq = DATA_DIR / "news_test_clean.parquet"
    if not train_pq.is_file():
        raise FileNotFoundError(
            f"Missing {train_pq}. Run day8 through the parquet export step first."
        )

    df_train = pd.read_parquet(train_pq)
    X_train = df_train[["text_clean", "text"]].copy()
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["label"])

    model = build_hybrid_pipeline()
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_s = time.time() - t0
    print(f"Fit on {len(df_train):,} docs in {fit_s:.1f}s")

    bundle = {"pipeline": model, "classes": le.classes_}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"Saved artifact: {args.out}")

    if args.skip_test_metrics or not test_pq.is_file():
        if not test_pq.is_file():
            print(f"(No {test_pq} — skipped test metrics.)")
        return

    df_test = pd.read_parquet(test_pq)
    if len(df_test) < 100:
        print(
            f"(Note: test parquet has only {len(df_test)} rows — metrics are not a full "
            "benchmark; use a full Day 8 test export for representative scores.)\n"
        )
    X_test = df_test[["text_clean", "text"]].copy()
    y_test = le.transform(df_test["label"])
    y_pred = model.predict(X_test)
    print("\nTest set")
    print(f"  Macro F1    : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"  Weighted F1 : {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(
        "\n",
        classification_report(
            y_test, y_pred, target_names=le.classes_, digits=3, zero_division=0
        ),
    )


if __name__ == "__main__":
    main()
