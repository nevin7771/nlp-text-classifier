# NLP News Classifier — 30-Day NLP Challenge (Days 8–12)

6-class news article categorisation using **spaCy + scikit-learn**.  
Part of the [30-day build-in-public NLP challenge](https://github.com/nevin7771/NLP).

---

## Project Overview

| Day | Notebook | Focus |
|-----|----------|-------|
| 8 | `day8_news_eda_preprocessing.ipynb` | EDA + spaCy preprocessing pipeline |
| 9 | `day9_baseline_tfidf.ipynb` | TF-IDF baseline + 5-fold cross-validation |
| 10 | `day10_feature_engineering.ipynb` | Hybrid pipeline: TF-IDF + linguistic features |
| 11 | `day11_final_model.ipynb` | Model persistence, `train.py`, `predict.py` |
| 12 | `day12_streamlit_app.ipynb` · `app.py` | Streamlit live demo + decision scores |

## Dataset

**20 Newsgroups** (sklearn built-in — no manual download needed)  
Filtered to 6 categories: `POLITICS · SCIENCE · HEALTH · SPORTS · TECHNOLOGY · LIFESTYLE`

## Setup

### 1. Install uv (if not already installed)
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtual environment and install dependencies
```bash
cd path/to/nlp-text-classifier
uv sync
# Jupyter + matplotlib/seaborn (Days 8–11 notebooks): add the notebooks extra
uv sync --extra notebooks
```

Python **3.11–3.13** is supported (`requires-python` caps below 3.14 so spaCy wheels stay available on Linux hosts such as **Streamlit Community Cloud**).

`en_core_web_sm` is installed from `pyproject.toml` (no separate `spacy download` required for the small model).

### 3. Optional: larger spaCy model (Day 10 notes)
```bash
uv run python -m spacy download en_core_web_md
```

### 4. Launch Jupyter
```bash
uv run jupyter lab
```
Then open `notebooks/day8_news_eda_preprocessing.ipynb`.

### 5. Register the uv venv as a Jupyter kernel (one-time)
```bash
uv run python -m ipykernel install --user --name nlp-news --display-name "Python (nlp-news)"
```

---

## Running the full pipeline (Day 11+)

```bash
# Train and save model
uv run python train.py

# CLI inference
echo "Scientists discover new exoplanets near distant stars" | uv run python predict.py

# Streamlit app
uv run streamlit run app.py
```

### Streamlit Community Cloud

1. **Python version:** In the app **Settings → Python version**, choose **3.12** (or **3.11** / **3.13**). Avoid prerelease Python versions: spaCy publishes wheels per CPython release; mismatches show up as “no wheel for the current platform”.
2. **Dependencies:** The repo uses `pyproject.toml` + `uv.lock`. After this project’s updates, the default install is slimmer (Jupyter is optional via `--extra notebooks`).
3. **Model file:** `models/news_hybrid.joblib` is **not** in Git. Either:
   - **Recommended:** Upload the file to **GitHub Releases** (or any static HTTPS URL), then in **App settings → Secrets** add:
     ```toml
     NEWS_CLASSIFIER_MODEL_URL = "https://github.com/USER/REPO/releases/download/TAG/news_hybrid.joblib"
     ```
     The app downloads it once to `/tmp` and caches it in session.
   - Or set **`NEWS_CLASSIFIER_MODEL_PATH`** to a path where your host mounts the file.

---

## Stack

`Python 3.11` · `spaCy 3.7` · `scikit-learn 1.4` · `pandas 2.1` · `Streamlit 1.30` · `joblib` · `uv`

---

## Results (Day 11)

| Metric | Score |
|--------|-------|
| Macro F1 (test) | ~0.89 |
| Best class | POLITICS (F1: ~0.96) |
| Hardest class | HEALTH / LIFESTYLE overlap |
| Training time | ~30s on CPU |
