# -*- coding: utf-8 -*-
"""
formulas.py - Feature Engineering for NEET Rank Predictor
==========================================================
All feature-engineering transformations live here.
Both train.py and predict.py derive features through this module.

Training vs Inference:
  - Training   : per-subject difficulties (physics/chemistry/biology) are
                 available in the CSV and used as real values.
  - Inference  : per-subject difficulties are unknown before the exam, so
                 build_feature_row() uses the overall difficulty_score for
                 all three section features (reasonable approximation).

Functions
---------
encode_categoricals(df)   -> df with encoded difficulty/paper columns
engineer_features(df)     -> df with all derived feature columns
prepare_dataframe(df)     -> convenience wrapper (encode + engineer)
build_feature_row(...)    -> single dict for inference (no DataFrame needed)

Constants
---------
DIFF_MAP        : str -> int  (Title-case, for training data)
DIFF_MAP_LOWER  : str -> int  (lowercase, for CLI input)
DIFF_LABEL      : int -> str  (reverse map)
"""

import pandas as pd

# ---------------------------------------------------------------
# Encoding maps
# ---------------------------------------------------------------
DIFF_MAP       = {"Easy": 1, "Moderate": 2, "Hard": 3}   # for training CSV
DIFF_MAP_LOWER = {"easy": 1, "moderate": 2, "hard": 3}   # for CLI input
DIFF_LABEL     = {1: "Easy", 2: "Moderate", 3: "Hard"}


# ---------------------------------------------------------------
# DataFrame-level transformations (used by train.py)
# ---------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical difficulty and paper_type columns into integers.
    Returns a new DataFrame.

    Columns added:
        physics_diff_enc    : int  (1=Easy, 2=Moderate, 3=Hard)
        chemistry_diff_enc  : int
        biology_diff_enc    : int
        paper_type_enc      : int  (0=Standard, 1=Controversial)
    """
    df = df.copy()
    df["physics_diff_enc"]   = df["physics_difficulty"].map(DIFF_MAP).fillna(2).astype(int)
    df["chemistry_diff_enc"] = df["chemistry_difficulty"].map(DIFF_MAP).fillna(2).astype(int)
    df["biology_diff_enc"]   = df["biology_difficulty"].map(DIFF_MAP).fillna(2).astype(int)
    df["paper_type_enc"]     = (df["paper_type"] == "Controversial").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived feature columns to the DataFrame.
    Call AFTER encode_categoricals().

    Features added:
        marks_pct              : marks / 720  (normalized score)
        score_gap_from_top     : top_score - marks
        marks_sq               : marks^2  (non-linear term)
        appeared_x_difficulty  : total_appeared_lakhs * difficulty_score
        marks_x_appeared       : marks * total_appeared_lakhs
        gap_x_difficulty       : score_gap_from_top * difficulty_score
        section_avg_difficulty : mean of per-subject encoded difficulties
    """
    df = df.copy()
    df["marks_pct"]              = df["marks"] / 720.0
    df["score_gap_from_top"]     = df["top_score"] - df["marks"]
    df["marks_sq"]               = df["marks"] ** 2
    df["appeared_x_difficulty"]  = df["total_appeared_lakhs"] * df["difficulty_score"]
    df["marks_x_appeared"]       = df["marks"] * df["total_appeared_lakhs"]
    df["gap_x_difficulty"]       = df["score_gap_from_top"] * df["difficulty_score"]
    df["section_avg_difficulty"] = (
        df["physics_diff_enc"] + df["chemistry_diff_enc"] + df["biology_diff_enc"]
    ) / 3.0
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper: encode categoricals then engineer features.
    Returns a new DataFrame ready for model.FEATURES selection.
    """
    df = encode_categoricals(df)
    df = engineer_features(df)
    return df


# ---------------------------------------------------------------
# Single-row inference (used by predict.py)
# ---------------------------------------------------------------
def build_feature_row(
    marks: int,
    appeared: float,
    difficulty: int,
    top_score: int,
    paper_type_enc: int = 0,
) -> dict:
    """
    Build one feature dict for a single inference call.
    Keys exactly match model.FEATURES (all 15 features).

    Per-subject difficulties (physics/chemistry/biology) are not
    available before a future exam, so the overall difficulty_score
    is used for all three section features -- consistent with the
    original input.py design.

    Parameters
    ----------
    marks          : NEET score (0-720)
    appeared       : total candidates appeared (in lakhs)
    difficulty     : difficulty encoded (1=Easy, 2=Moderate, 3=Hard)
    top_score      : estimated highest score for the year
    paper_type_enc : 0=Standard, 1=Controversial (default 0)
    """
    gap = top_score - marks
    return {
        "marks":                  marks,
        "total_appeared_lakhs":   appeared,
        "difficulty_score":       difficulty,
        "top_score":              top_score,
        "marks_pct":              marks / 720.0,
        "score_gap_from_top":     gap,
        "marks_sq":               marks ** 2,
        "appeared_x_difficulty":  appeared * difficulty,
        "marks_x_appeared":       marks * appeared,
        "gap_x_difficulty":       gap * difficulty,
        # Per-subject unknown pre-exam → approximate with overall difficulty
        "section_avg_difficulty": float(difficulty),
        "physics_diff_enc":       difficulty,
        "chemistry_diff_enc":     difficulty,
        "biology_diff_enc":       difficulty,
        "paper_type_enc":         paper_type_enc,
    }