@echo off
REM ============================================================
REM  NLP News Classifier — One-click Git + uv setup for Windows
REM  Run this from the project folder in CMD or Git Bash
REM ============================================================

echo.
echo ====================================================
echo  Step 1: Git setup
echo ====================================================

git init -b main
git config user.name "naveen"
git config user.email "awsnaveen10@gmail.com"

echo.
echo ====================================================
echo  Step 2: Initial commit
echo ====================================================

git add .
git commit -m "Day 8: NLP News Classifier project setup

- 6-class news categorisation (20 Newsgroups filtered to 6 categories)
- spaCy + scikit-learn stack
- SpacyPreprocessor + SpacyFeatureExtractor utilities (src/nlp_utils_news.py)
- Day 8 EDA + preprocessing notebook
- uv project config (pyproject.toml)
- .gitignore, README, runtime.txt

Part of 30-day NLP build-in-public challenge (Days 8-12)"

echo.
echo ====================================================
echo  Step 3: uv environment setup
echo ====================================================

uv sync

echo.
echo ====================================================
echo  Step 4: Install spaCy models
echo ====================================================

uv run python -m spacy download en_core_web_sm
uv run python -m spacy download en_core_web_md

echo.
echo ====================================================
echo  Step 5: Register Jupyter kernel
echo ====================================================

uv run python -m ipykernel install --user --name nlp-news --display-name "Python (nlp-news)"

echo.
echo ====================================================
echo  Setup complete!
echo ====================================================
echo.
echo  To add GitHub remote and push:
echo    git remote add origin https://github.com/nevin7771/NLP.git
echo    git push -u origin main
echo.
echo  OR create a new repo:
echo    gh repo create nlp-news-classifier --public --source=. --push
echo.
echo  To launch Jupyter:
echo    uv run jupyter lab
echo.
pause
