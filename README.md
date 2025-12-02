# Virality — UCI Online News Popularity analysis

A compact project that explores and models article virality using the UCI Online News Popularity dataset (~39k articles, ~61 features). The repository contains scripts and notebooks for data cleaning, feature engineering, EDA, clustering, and model experiments (baseline and neural models).

## TL;DR
- Run data processing to produce cleaned datasets.
- Run EDA and clustering scripts or open the notebooks for experiments.
- Notebooks contain model training/evaluation and produced artifacts (CSV results and a tuned Random Forest pickle).

## Open the notebooks for modeling and experiments:
	- `step3_feature_engineering.ipynb` — feature engineering
	- `step5_baseline_models.ipynb` — baseline model experiments
	- `step7_neural_models.ipynb` — neural model experiments

## Repository layout (important files)
- `main.py` / `src/step1_data_cleaning.py` — data cleaning and light feature engineering; writes processed CSVs.
- `src/EDA.py` — exploratory analysis and plotting (writes `EDA_summary_output.txt` and figures).
- `src/step6_clustering.py` — clustering pipeline (KMeans), saves `features_with_clusters.csv` and cluster metrics and figures.
- `data/raw/OnlineNewsPopularity.csv` — original dataset.
- `data/processed/` — processed datasets and outputs (e.g., `cleaned_with_features.csv`, `features_complete.csv`, `features_with_clusters.csv`, `cluster_metrics.csv`).
- `step3_feature_engineering.ipynb`, `step5_baseline_models.ipynb`, `step7_neural_models.ipynb` — notebooks for engineering and modeling.
- `baseline_model_results.csv`, `tuned_random_forest.pkl` — example experiment outputs / model artifact.

## Notes & gotchas
- Viral label: defined as articles in the top 10% of `shares` (90th percentile). The code currently computes a `shares_log` via log1p and a binary `viral` flag.
- Column names in the original CSV contain leading spaces (e.g., `" shares"`) — scripts reference these names. If you rename/clean column names, update scripts accordingly or normalize names early (recommended).
- `src/step6_clustering.py` expects `features_complete.csv` (PCA components for headline text like `headline_pc*` plus engineered features). If that file is missing, the script raises a FileNotFoundError.

.

## Contact
For questions about the project, open an issue or reach out to the repo owner.

---
Generated: concise one-page README to help run and understand the pipeline.
