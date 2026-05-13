# -*- coding: utf-8 -*-
"""
NEET 2026 Rank Predictor - Training Pipeline
=============================================
Trains an ensemble model on ALL historical NEET data (2021-2025),
evaluates with cross-year leave-one-out validation,
and saves the fitted pipeline to disk for use by predict.py.

Strategy:
  - Train on all years (2021-2025) so the model sees Hard-paper data (2025).
  - Use Leave-One-Group-Out CV to report honest year-wise errors.
  - Final model trained on the full dataset before saving.

Usage:
    python train.py
"""

import io
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from model import build_ensemble, FEATURES, TARGET, YEAR_WEIGHTS
from formulas import prepare_dataframe

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "neet_rank_prediction_dataset.csv")

print("=" * 60)
print("  NEET 2026 Rank Predictor - Model Training")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"\n[OK] Dataset loaded: {len(df)} rows, years {df['year'].unique().tolist()}")

# ---------------------------------------------------------------
# 2 & 3. Encoding + feature engineering (via formulas.py)
# ---------------------------------------------------------------
df = prepare_dataframe(df)   # encode categoricals + add derived features

# ---------------------------------------------------------------
# 4. Sample weights
# ---------------------------------------------------------------
df["sample_weight"] = df["year"].map(YEAR_WEIGHTS).fillna(1.0)

# ---------------------------------------------------------------
# 5. Leave-One-Year-Out cross-validation (honest evaluation)
# ---------------------------------------------------------------
try:
    import xgboost  # noqa: F401
    USE_XGB = True
    print("[OK] XGBoost available - included in ensemble")
except ImportError:
    USE_XGB = False
    print("[!!] XGBoost not found - sklearn-only ensemble")

print("\n[..] Running Leave-One-Year-Out cross-validation...")

X      = df[FEATURES]
y      = np.log1p(df[TARGET])
groups = df["year"]
weights = df["sample_weight"]

logo        = LeaveOneGroupOut()
cv_results  = []

for train_idx, test_idx in logo.split(X, y, groups):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr       = y.iloc[train_idx]
    w_tr       = weights.iloc[train_idx]
    y_te_raw   = df[TARGET].iloc[test_idx]   # raw ranks for eval
    year_val   = groups.iloc[test_idx].iloc[0]

    mdl = build_ensemble(use_xgb=USE_XGB)
    mdl.fit(X_tr, y_tr, sample_weight=w_tr)

    y_pred = np.expm1(mdl.predict(X_te))
    y_pred = np.clip(y_pred, 1, None)

    mape = mean_absolute_percentage_error(y_te_raw, y_pred) * 100
    mae  = mean_absolute_error(y_te_raw, y_pred)
    cv_results.append({"year": year_val, "MAPE%": round(mape, 1), "MAE": round(mae, 0)})

cv_df = pd.DataFrame(cv_results).sort_values("year")
print("\n-- Leave-One-Year-Out CV results -----------------------------")
print(f"  {'Year':>6}  {'MAPE%':>8}  {'MAE (ranks)':>12}")
print("  " + "-" * 32)
for _, r in cv_df.iterrows():
    note = "  <- anomalous paper" if r["year"] == 2024 else ""
    print(f"  {int(r['year']):>6}  {r['MAPE%']:>7.1f}%  {int(r['MAE']):>12,}{note}")

avg_mape = cv_df[cv_df["year"] != 2024]["MAPE%"].mean()
print(f"\n  Avg MAPE (excl. 2024 anomaly): {avg_mape:.1f}%")

# ---------------------------------------------------------------
# 6. Train FINAL model on all data
# ---------------------------------------------------------------
print("\n[..] Training final model on all years (2021-2025)...")
final_model = build_ensemble(use_xgb=USE_XGB)
final_model.fit(X, y, sample_weight=weights)
print("[OK] Final model trained")

# Quick in-sample sanity check
y_pred_all = np.expm1(final_model.predict(X))
y_pred_all = np.clip(y_pred_all, 1, None).round().astype(int)
mape_all   = mean_absolute_percentage_error(df[TARGET], y_pred_all) * 100
mae_all    = mean_absolute_error(df[TARGET], y_pred_all)
r2_all     = r2_score(y, np.log1p(y_pred_all))

print(f"\n-- In-sample fit (full dataset) ------------------------------")
print(f"   MAPE : {mape_all:.2f}%")
print(f"   MAE  : {mae_all:,.0f} ranks")
print(f"   R2   : {r2_all:.4f}")

# ---------------------------------------------------------------
# 7. Save model + metadata
# ---------------------------------------------------------------
MODEL_PATH = os.path.join(SCRIPT_DIR, "neet_model.pkl")
meta = {
    "model":           final_model,
    "features":        FEATURES,
    "cv_mape_avg":     round(avg_mape, 2),
    "cv_results":      cv_df.to_dict("records"),
    "insample_mape":   round(mape_all, 2),
    "insample_mae":    round(mae_all, 0),
    "use_xgb":         USE_XGB,
}
joblib.dump(meta, MODEL_PATH)
print(f"\n[OK] Model saved -> {MODEL_PATH}")
print(f"     CV MAPE (ex-2024): {avg_mape:.1f}% | trained on {len(df)} samples")
print("     Run  python predict.py  to make predictions for 2026.\n")
