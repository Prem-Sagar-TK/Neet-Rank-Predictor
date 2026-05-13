"""
model.py — Core model definitions and feature schema
for the NEET 2026 Rank Predictor.

This module is imported by train.py and predict.py.
It does NOT train anything by itself; run train.py to fit the model.
"""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor

# ─────────────────────────────────────────────────────────────
# Feature schema (must match training and inference)
# ─────────────────────────────────────────────────────────────
FEATURES = [
    # Raw inputs
    "marks",
    "total_appeared_lakhs",
    "difficulty_score",
    "top_score",
    # Engineered features (see formulas.py)
    "marks_pct",
    "score_gap_from_top",
    "marks_sq",
    "appeared_x_difficulty",
    "marks_x_appeared",
    "gap_x_difficulty",
    "section_avg_difficulty",
    # Section-wise encoded difficulties
    "physics_diff_enc",
    "chemistry_diff_enc",
    "biology_diff_enc",
    # Paper type flag
    "paper_type_enc",
]

TARGET = "air_rank"

# ─────────────────────────────────────────────────────────────
# Sample weights: down-weight the anomalous 2024 paper
# (67 candidates scored 720/720 due to grace marks controversy)
# ─────────────────────────────────────────────────────────────
YEAR_WEIGHTS = {
    2021: 1.0,
    2022: 1.0,
    2023: 1.2,
    2024: 0.3,   # anomalous year — heavily down-weighted
    2025: 1.5,   # most recent, most relevant
}

# ─────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────
def build_ensemble(use_xgb: bool = True) -> VotingRegressor:
    """
    Returns an untrained VotingRegressor ensemble.

    Components:
      - GradientBoostingRegressor  (high accuracy, handles non-linearity well)
      - RandomForestRegressor      (robust baseline, variance reduction)
      - XGBRegressor               (fastest + best for tabular, if available)

    The target variable (air_rank) must be log-transformed before fitting:
        y_train_log = np.log1p(y_train)
    and back-transformed after prediction:
        y_pred = np.expm1(model.predict(X))
    """
    gb = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        min_samples_leaf=3,
        random_state=42,
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    estimators = [("gb", gb), ("rf", rf)]

    if use_xgb:
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
            estimators.append(("xgb", xgb_model))
        except ImportError:
            pass

    return VotingRegressor(estimators=estimators)
