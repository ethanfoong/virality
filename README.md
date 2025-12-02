# Virality — UCI Online News Popularity analysis

This repository implements a reproducible pipeline for exploring and modeling article "virality" using the UCI Online News Popularity dataset (~39k articles). It includes data cleaning, deterministic feature engineering (text + metadata), EDA, clustering, baseline and ensemble models, and lightweight neural models. Artifacts and figures are written to `data/processed/`, `models/`, and `figures/` so results can be inspected independently of training.

## What this repo contains (high level)
- Data ingestion and cleaning: scripts/notebooks normalize column names, deduplicate, and produce `cleaned_base.csv` and `cleaned_with_features.csv` under `data/processed/`.
- Feature engineering: headline length, keyword density, headline TF-IDF -> PCA components, sentiment where available, and other deterministic transforms.
- Modeling: baseline linear models, a Random Forest ensemble (tuned grid), and an optional Keras MLP regressor. Model artifacts are saved under `models/`.
- Clustering: KMeans over engineered numeric + headline PC features; cluster labels and metrics saved in `data/processed/`.
- Prediction & analysis: a Step 8 notebook containing a `predict_from_dict()` API and scenario simulations; plus a new `analysis/run_analysis.py` script that produces quick evaluation plots and a short markdown report.

## Quick findings (summary)
- Defining virality: articles in the top 10% by `shares` (90th percentile) are labeled `viral`.
- Feature signal: metadata (article length, number of keywords / keyword density) and headline-derived features (TF-IDF PCA components, headline length) are consistently informative across experiments.
- Modelling: tree-based ensembles (Random Forest) provide a robust baseline for predicting log-shares (`shares_log`) and give interpretable feature importances; a small MLP can serve as a useful fallback when non-linear interactions are important.
- Caveat: numeric performance varies by preprocessing and feature selection. Large binary/text assets are handled with Git LFS to keep the repo size manageable.

## How to reproduce key outputs
1. Install dependencies listed in `requirements.txt` (create a fresh venv/conda env first).
2. Make sure data is present under `data/raw/OnlineNewsPopularity.csv` or fetch it from the UCI repository.
3. Run the main notebook or scripts in order (recommended):

```powershell
# from project root (PowerShell)
python -m pip install -r requirements.txt
# run the primary notebook top-to-bottom or call processing scripts
# quick: run the analysis helper to validate artifacts and generate plots
python analysis/run_analysis.py
```

Generated files and locations:
- Processed data: `data/processed/cleaned_with_features.csv`, `data/processed/features_complete.csv`, `data/processed/features_with_clusters.csv`
- Models: `models/best_random_forest.joblib`, `models/mlp_regressor.keras`, scalers and vectorizers under `models/`
- Figures: `figures/` (EDA, SHAP summary if available, prediction vs actual, residuals, scenario simulations)
- Analysis report: `analysis/report.md`

## Step 8 and prediction API
Step 8 (`step8_final_integration.ipynb`) provides a consolidated prediction utility, `predict_from_dict()`, which loads available artifacts (TF-IDF/PCA, RF, MLP) and returns predicted shares and a viral probability estimate. The notebook also includes scenario simulations (headline edits, keyword counts, etc.) and dataset-level plotting helpers.

## Notes & gotchas
- Git LFS: this repo uses Git LFS for large artifacts (e.g., `data/processed/features_complete.csv` may be stored via LFS). If you see pointer file warnings, run:

```powershell
git lfs install
git lfs pull
```

- Column naming: original CSV has some columns with leading spaces; the pipeline normalizes names but if you write custom scripts, ensure consistent normalization.
- Re-running training may overwrite model artifacts in `models/` — back up any models you need.

## Next steps & suggestions
- Run `analysis/run_analysis.py` to generate a short evaluation report and plots.
- If you want an HTML report or automated CI checks, I can add a small GitHub Actions workflow to run the analysis and publish artifacts.

## Contact
Open an issue or email ethanfoong@berkeley.edu for questions or to share improvements.

