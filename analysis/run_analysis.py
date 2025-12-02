"""analysis/run_analysis.py

Creates a small analysis report and figures using existing artifacts in the repo.
Run: python analysis/run_analysis.py

Outputs:
 - figures/analysis_pred_vs_actual.png
 - figures/analysis_residuals.png
 - figures/analysis_feature_importances.png
 - analysis/report.md

The script is defensive: it will report missing artifacts instead of failing.
"""

from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
FIGURES = PROJECT_ROOT / 'figures'
ANALYSIS_DIR = PROJECT_ROOT / 'analysis'

for d in [FIGURES, ANALYSIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPORT_PATH = ANALYSIS_DIR / 'report.md'


def safe_load_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"Could not read {p}: {e}")
        return None


def load_artifacts():
    artifacts = {}
    import joblib
    # dataset
    features_p = DATA_PROCESSED / 'features_complete.csv'
    artifacts['features_complete'] = safe_load_csv(features_p) if features_p.exists() else None
    # rf
    rf_p = MODELS_DIR / 'best_random_forest.joblib'
    if rf_p.exists():
        try:
            artifacts['rf'] = joblib.load(rf_p)
        except Exception as e:
            print('Failed to load RF model:', e)
            artifacts['rf'] = None
    else:
        artifacts['rf'] = None
    # mlp scaler
    scaler_p = MODELS_DIR / 'mlp_scaler.joblib'
    if scaler_p.exists():
        try:
            artifacts['mlp_scaler'] = joblib.load(scaler_p)
        except Exception as e:
            print('Failed to load scaler:', e)
            artifacts['mlp_scaler'] = None
    else:
        artifacts['mlp_scaler'] = None
    # tf/keras model
    mlp_model_p = MODELS_DIR / 'mlp_regressor.keras'
    if mlp_model_p.exists():
        try:
            import tensorflow as tf
            artifacts['mlp_model'] = tf.keras.models.load_model(mlp_model_p)
        except Exception as e:
            print('Failed to load MLP model:', e)
            artifacts['mlp_model'] = None
    else:
        artifacts['mlp_model'] = None

    return artifacts


def metrics_for_predictions(y_true, y_pred):
    # y_true and y_pred are in original shares scale
    # compute metrics on log1p scale and raw scale
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(np.clip(y_pred, 0, None))
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true_log, y_pred_log)
    return {'rmse_log': float(rmse_log), 'mae': float(mae), 'r2_log': float(r2)}


def evaluate_rf(artifacts):
    rf = artifacts.get('rf')
    df = artifacts.get('features_complete')
    if rf is None or df is None:
        print('Skipping RF evaluation: missing model or dataset')
        return None
    # determine features
    if hasattr(rf, 'feature_names_in_'):
        features = list(rf.feature_names_in_)
    else:
        features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['shares', 'shares_log', 'viral']]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print('Warning: RF expects features not present in dataset:', missing[:10])
        # add missing columns as zeros
        for f in missing:
            df[f] = 0
    X = df[features].fillna(0)
    try:
        preds_log = rf.predict(X)
    except Exception as e:
        print('RF prediction failed:', e)
        return None
    preds = np.expm1(preds_log)
    df['pred_shares_rf'] = preds
    m = metrics_for_predictions(df['shares'].values, df['pred_shares_rf'].values)
    # plots
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=np.log1p(df['shares']), y=np.log1p(df['pred_shares_rf']), alpha=0.4)
    plt.xlabel('Actual shares (log1p)')
    plt.ylabel('Predicted shares (log1p)')
    plt.title('RF: Predicted vs Actual (log1p)')
    plt.tight_layout()
    out1 = FIGURES / 'analysis_pred_vs_actual.png'
    plt.savefig(out1)
    plt.close()

    plt.figure(figsize=(8,4))
    df['residual_rf'] = np.log1p(df['pred_shares_rf']) - np.log1p(df['shares'])
    sns.histplot(df['residual_rf'].dropna(), bins=80)
    plt.title('RF residuals (log scale)')
    out2 = FIGURES / 'analysis_residuals.png'
    plt.savefig(out2)
    plt.close()

    # feature importances
    try:
        importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
        top = importances.head(30)
        plt.figure(figsize=(8,6))
        sns.barplot(x=top.values, y=top.index)
        plt.title('RF feature importances (top 30)')
        out3 = FIGURES / 'analysis_feature_importances.png'
        plt.tight_layout()
        plt.savefig(out3)
        plt.close()
    except Exception as e:
        print('Could not compute RF importances:', e)
        out3 = None

    return {'metrics': m, 'plots': {'pred_vs_actual': str(out1), 'residuals': str(out2), 'feature_importances': str(out3) if out3 else None}}


def evaluate_mlp(artifacts):
    model = artifacts.get('mlp_model')
    scaler = artifacts.get('mlp_scaler')
    df = artifacts.get('features_complete')
    if model is None or scaler is None or df is None:
        print('Skipping MLP evaluation: missing model, scaler, or dataset')
        return None
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['shares', 'shares_log', 'viral']]
    X = df[features].fillna(0)
    try:
        Xs = scaler.transform(X)
    except Exception as e:
        print('Scaler transform failed:', e)
        return None
    try:
        preds_log = model.predict(Xs).reshape(-1)
    except Exception as e:
        print('MLP prediction failed:', e)
        return None
    preds = np.expm1(preds_log)
    df['pred_shares_mlp'] = preds
    m = metrics_for_predictions(df['shares'].values, df['pred_shares_mlp'].values)
    return {'metrics': m}


def write_report(results):
    lines = ['# Analysis Report', '', f'Date: {pd.Timestamp.now()}', '']
    if results.get('rf'):
        rf_r = results['rf']
        lines += ['## Random Forest Evaluation', '']
        lines += [f"- RMSE (log1p): {rf_r['metrics']['rmse_log']:.4f}", f"- MAE (raw): {rf_r['metrics']['mae']:.2f}", f"- R2 (log1p): {rf_r['metrics']['r2_log']:.4f}", '']
        plots = rf_r.get('plots', {})
        if plots.get('pred_vs_actual'):
            lines += [f"Pred vs Actual plot: `{plots['pred_vs_actual']}`"]
        if plots.get('residuals'):
            lines += [f"Residuals plot: `{plots['residuals']}`"]
        if plots.get('feature_importances'):
            lines += [f"Feature importances: `{plots['feature_importances']}`"]
        lines += ['']
    if results.get('mlp'):
        mlp_r = results['mlp']
        lines += ['## MLP Evaluation', '']
        lines += [f"- RMSE (log1p): {mlp_r['metrics']['rmse_log']:.4f}", f"- MAE (raw): {mlp_r['metrics']['mae']:.2f}", f"- R2 (log1p): {mlp_r['metrics']['r2_log']:.4f}", '']
    # missing artifacts
    lines += ['## Missing artifacts / notes', '']
    missing = results.get('missing', [])
    if missing:
        for m in missing:
            lines += [f'- {m}']
    else:
        lines += ['- None']
    # write file
    REPORT_PATH.write_text('\n'.join(lines))
    print('Wrote analysis report to', REPORT_PATH)


def main():
    artifacts = load_artifacts()
    missing = []
    if artifacts.get('features_complete') is None:
        missing.append('features_complete.csv')
    if artifacts.get('rf') is None:
        missing.append('best_random_forest.joblib')
    if artifacts.get('mlp_model') is None and artifacts.get('mlp_scaler') is None:
        missing.append('mlp_regressor.keras or mlp_scaler.joblib')

    results = {'missing': missing}

    rf_res = evaluate_rf(artifacts)
    if rf_res:
        results['rf'] = rf_res
    mlp_res = evaluate_mlp(artifacts)
    if mlp_res:
        results['mlp'] = mlp_res

    write_report(results)
    print('Done')


if __name__ == '__main__':
    main()
