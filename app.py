#!/usr/bin/env python3
"""Day 12 — Streamlit demo for the 6-class news hybrid model."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import load_artifact, predict_with_details
from src.model_resolve import resolve_model_file

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


def _secret_model_url() -> str:
    try:
        u = st.secrets.get("NEWS_CLASSIFIER_MODEL_URL", "")
        if isinstance(u, str):
            return u.strip()
        if u is not None:
            return str(u).strip()
    except Exception:
        pass
    return ""


def _merged_model_url() -> str:
    """Env wins (Streamlit may mirror secrets into env on some hosts)."""
    u = os.environ.get("NEWS_CLASSIFIER_MODEL_URL", "").strip()
    if u:
        return u
    return _secret_model_url()


def _merged_force_download() -> bool:
    v = os.environ.get("NEWS_CLASSIFIER_FORCE_DOWNLOAD", "").strip().lower()
    if v in ("1", "true", "yes"):
        return True
    try:
        s = st.secrets.get("NEWS_CLASSIFIER_FORCE_DOWNLOAD", None)
        if s is True:
            return True
        if isinstance(s, str) and s.strip().lower() in ("1", "true", "yes"):
            return True
    except Exception:
        pass
    return False


@st.cache_resource
def _cached_bundle(path_str: str):
    return load_artifact(Path(path_str))


def _inject_sample() -> None:
    choice = st.session_state.get("sample_pick", "(none)")
    if choice != "(none)" and choice in SAMPLES:
        st.session_state["main_text"] = SAMPLES[choice]


def _deploy_help() -> None:
    st.markdown(
        r"""
**Why this happens:** `models/news_hybrid.joblib` is not in Git (too large / generated).

**If you see HTTP 404:** GitHub has **no file** at that URL yet. Example URLs in docs are **templates** — you must create the release and upload the asset first.

**Create the release (one-time):**

1. Train locally: `uv run python train.py` → produces `models/news_hybrid.joblib`.
2. On GitHub: **Repo → Releases → Create a new release**.
3. Choose a **new tag** (e.g. `model-v1` or `v1.0.0`) and publish.
4. **Attach** `news_hybrid.joblib` under “Attach binaries” (exact name can differ; the URL will match what you upload).
5. After publishing, open the release page, **right‑click the file** → **Copy link** (or click Download and copy the URL from the address bar).
6. Paste **only that URL** into Streamlit **Secrets**:
   ```toml
   NEWS_CLASSIFIER_MODEL_URL = "https://github.com/OWNER/REPO/releases/download/TAG/FILENAME.joblib"
   ```
7. Save secrets, **Restart** the app, and set `NEWS_CLASSIFIER_FORCE_DOWNLOAD = true` once if it cached a bad download.

**Test:** Open the URL in a private browser tab — it must **download** a multi‑MB file, not show a GitHub 404 page.

**Locally:** run `uv run python train.py`, then `uv run streamlit run app.py` (no Secrets needed).
        """
    )


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
        "Cloud: set `NEWS_CLASSIFIER_MODEL_URL` in Secrets (see sidebar / expander below)."
    )

    with st.sidebar:
        st.subheader("Model")
        override = st.text_input(
            "Optional: path to .joblib (overrides env)",
            value="",
            placeholder="Leave empty to use env / default / download URL",
            help="Local path only. On Cloud, use Secrets NEWS_CLASSIFIER_MODEL_URL instead.",
        )
        if _merged_model_url():
            st.info("Using **NEWS_CLASSIFIER_MODEL_URL** (env or Secrets).")
        elif os.environ.get("NEWS_CLASSIFIER_MODEL_PATH"):
            st.info("Using **NEWS_CLASSIFIER_MODEL_PATH**.")
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

    try:
        with st.spinner("Loading model (downloading on first run may take a minute)…"):
            path = resolve_model_file(
                override if override.strip() else None,
                model_url=_merged_model_url() or None,
                force_download=_merged_force_download(),
            )
            bundle = _cached_bundle(str(path))
    except (FileNotFoundError, ValueError, OSError) as e:
        st.error(str(e))
        with st.expander("How to fix (especially on Streamlit Cloud)", expanded=True):
            _deploy_help()
        st.stop()
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
