#!/usr/bin/env python3
"""Day 12 — Streamlit demo for the 6-class news hybrid model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import default_artifact_path, load_artifact, predict_with_details

SAMPLES = {
    "Science": (
        "NASA researchers announced new findings from the James Webb Space Telescope "
        "on atmospheric composition of a distant exoplanet orbiting a red dwarf."
    ),
    "Politics": (
        "The Senate debated a bipartisan infrastructure bill amid ongoing negotiations "
        "over funding for highways and rural broadband expansion."
    ),
    "Sports": (
        "The championship final went to overtime as both teams traded goals in a "
        "thrilling end-to-end match watched by millions."
    ),
    "Health": (
        "A large clinical trial found that the new vaccine candidate reduced severe "
        "disease outcomes in adults over 65 with multiple comorbidities."
    ),
    "Technology": (
        "The chipmaker unveiled its next-generation processor with improved AI "
        "acceleration and lower power draw for laptop workstations."
    ),
    "Lifestyle": (
        "Travel writers shared tips for slow tourism in small coastal towns, focusing "
        "on local food markets and weekend farmers' markets."
    ),
}


@st.cache_resource
def _cached_bundle(path_str: str):
    return load_artifact(Path(path_str))


def _inject_sample() -> None:
    choice = st.session_state.get("sample_pick", "(none)")
    if choice != "(none)" and choice in SAMPLES:
        st.session_state["main_text"] = SAMPLES[choice]


def main() -> None:
    st.set_page_config(
        page_title="News classifier",
        page_icon="📰",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.title("6-class news classifier")
    st.caption(
        "Hybrid model: TF-IDF + spaCy linguistic features · Day 12 demo · "
        "Train with `uv run python train.py` if the artifact is missing."
    )

    default_path = default_artifact_path()
    with st.sidebar:
        st.subheader("Model")
        artifact_path = st.text_input(
            "Path to joblib bundle",
            value=str(default_path),
            help="Default: models/news_hybrid.joblib after training.",
        )
        st.divider()
        st.subheader("Try a sample")
        st.selectbox(
            "Load example text",
            ["(none)"] + list(SAMPLES.keys()),
            key="sample_pick",
            on_change=_inject_sample,
        )
        st.divider()
        st.markdown(
            "**Classes:** HEALTH · LIFESTYLE · POLITICS · SCIENCE · SPORTS · TECHNOLOGY"
        )

    path = Path(artifact_path)
    if not path.is_file():
        st.error(
            f"Model not found at `{path}`. Run **`uv run python train.py`** from the "
            "project root, then refresh this page."
        )
        st.stop()

    try:
        bundle = _cached_bundle(str(path.resolve()))
    except Exception as e:
        st.exception(e)
        st.stop()

    if "main_text" not in st.session_state:
        st.session_state.main_text = ""

    text = st.text_area(
        "Paste news-style article text",
        height=220,
        placeholder="Paste a paragraph or more of English news-like text…",
        key="main_text",
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        run = st.button("Classify", type="primary", use_container_width=True)
    with col_b:
        if st.button("Clear", use_container_width=True):
            st.session_state.main_text = ""
            if "sample_pick" in st.session_state:
                st.session_state.sample_pick = "(none)"
            st.rerun()

    if not run:
        return

    if not text.strip():
        st.warning("Add some text to classify.")
        return

    with st.spinner("Running model…"):
        label, classes, scores = predict_with_details(bundle, text.strip())

    st.success(f"**Predicted category:** `{label}`")

    if scores is not None and len(scores) == len(classes):
        chart_df = pd.DataFrame({"class": classes, "score": scores}).sort_values(
            "score", ascending=False
        )
        st.subheader("Decision scores (higher = stronger margin for that class)")
        st.bar_chart(chart_df.set_index("class")["score"], horizontal=True)
        with st.expander("Raw scores"):
            st.dataframe(chart_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
