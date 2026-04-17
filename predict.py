#!/usr/bin/env python3
"""CLI: raw news text -> predicted class (Day 11)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.inference import load_artifact, predict_one


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict news category from raw text (stdin or argument)."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Raw article text. If omitted, reads stdin (e.g. pipe or heredoc).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to joblib bundle (default: models/news_hybrid.joblib)",
    )
    args = parser.parse_args()

    if args.text is not None:
        raw = args.text
    else:
        raw = sys.stdin.read()
    raw = raw.strip()
    if not raw:
        print("No input text.", file=sys.stderr)
        sys.exit(1)

    bundle = load_artifact(args.model)
    label = predict_one(bundle, raw)
    print(label)


if __name__ == "__main__":
    main()
