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
| 12 | Streamlit app | Live demo deployment |

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
cd Downloads/Projects/nlp-news-classifier
uv sync
```

### 3. Install spaCy language model
```bash
uv run python -m spacy download en_core_web_sm
# Day 10 also needs:
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
