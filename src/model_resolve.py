"""
Resolve path to ``news_hybrid.joblib`` for local runs and Streamlit Cloud.

Priority:
1. ``NEWS_CLASSIFIER_MODEL_PATH`` — absolute path on disk (e.g. mounted secret file)
2. ``model_url`` — passed in from the app (env + Streamlit Secrets merged there)
3. Default: ``models/news_hybrid.joblib`` next to the repo root
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

from src.inference import default_artifact_path

CACHE_DIR = Path(os.environ.get("NEWS_CLASSIFIER_CACHE_DIR", "/tmp/nlp_news_classifier"))
DEFAULT_DOWNLOAD_NAME = "news_hybrid.joblib"

# Reject README-style placeholders that users paste by mistake
_PLACEHOLDER_MARKERS = ("<tag>", "<TAG>", "YOUR_", "USER/REPO", "vX.Y.Z")


def _validate_url(url: str) -> None:
    u = url.strip()
    if not u.startswith("http://") and not u.startswith("https://"):
        raise ValueError(
            f"Model URL must start with https:// (got: {u[:80]!r}…)"
        )
    for marker in _PLACEHOLDER_MARKERS:
        if marker in u:
            raise ValueError(
                f"Model URL still contains a placeholder ({marker!r}). "
                "Create a GitHub Release, upload `news_hybrid.joblib`, then paste the "
                "**real** download link, e.g. "
                "`https://github.com/nevin7771/nlp-text-classifier/releases/download/v1.0.0/news_hybrid.joblib` "
                "(your tag can differ)."
            )


def download_model(url: str, dest: Path) -> None:
    _validate_url(url)
    req = urllib.request.Request(
        url.strip(),
        headers={"User-Agent": "nlp-news-classifier/1.0 (Streamlit)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            if code != 200:
                raise OSError(f"HTTP {code} when downloading model (check the URL in Secrets).")
            data = resp.read()
    except urllib.error.HTTPError as e:
        raise OSError(
            f"HTTP {e.code} for this URL — often a wrong path or missing release file. "
            f"Open the URL in a browser; it should start a download of the .joblib file.\nURL: {url[:200]}"
        ) from e
    except urllib.error.URLError as e:
        raise OSError(f"Could not reach model URL: {e}") from e

    head = data[:200].lstrip()
    if head.startswith((b"<!DOCTYPE", b"<html", b"<HTML")) or b"<title>Not Found" in data[:4000]:
        raise OSError(
            "Download looks like an HTML error page (often GitHub 404). "
            "Use a **direct** asset URL from **Releases → your tag → news_hybrid.joblib**, "
            "not a placeholder like `<tag>`."
        )
    if len(data) < 2000:
        raise OSError(
            f"Download is only {len(data)} bytes — not a valid model file. "
            "Fix `NEWS_CLASSIFIER_MODEL_URL` in Secrets."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def resolve_model_file(
    path_override: str | None,
    *,
    model_url: str | None = None,
    force_download: bool = False,
) -> Path:
    """Return a path to an existing ``.joblib`` file."""
    if path_override and str(path_override).strip():
        p = Path(path_override.strip()).expanduser()
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"Model path not found: {p}")

    env_path = os.environ.get("NEWS_CLASSIFIER_MODEL_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"NEWS_CLASSIFIER_MODEL_PATH is set but file missing: {p}")

    url = (model_url or "").strip()
    if url:
        dest = CACHE_DIR / DEFAULT_DOWNLOAD_NAME
        if force_download or not dest.is_file():
            download_model(url, dest)
        if dest.is_file() and dest.stat().st_size > 2000:
            return dest.resolve()
        raise FileNotFoundError(f"Download failed or incomplete: {dest}")

    default = default_artifact_path()
    if default.is_file():
        return default.resolve()

    raise FileNotFoundError(
        "No model artifact found. Train locally (`uv run python train.py`), "
        "or set NEWS_CLASSIFIER_MODEL_URL in Streamlit Secrets with a **direct HTTPS link** "
        "to the `.joblib` file (see README)."
    )
