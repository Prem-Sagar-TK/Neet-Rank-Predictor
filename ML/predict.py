# -*- coding: utf-8 -*-
"""
NEET 2026 Rank Predictor - Interactive CLI
==========================================
Predicts AIR rank for NEET 2026 given a student's score
and 2026 exam parameters.

Inputs required:
  - Your NEET marks
  - Overall paper difficulty  (Easy / Moderate / Hard)
  - Total candidates appeared (lakhs)
  - Estimated top score

Per-subject difficulties are NOT asked -- they're unknown before
the result is declared, so the model uses the overall difficulty
for all three sections (see formulas.py).

Usage:
    python predict.py                       # interactive mode
    python predict.py --score 580           # quick single prediction
    python predict.py --score 580 --difficulty Hard --appeared 25.0
    python predict.py --batch               # rank table for 420-720
"""

import io
import os
import sys
import argparse
import warnings
import numpy as np
import joblib
from formulas import build_feature_row, DIFF_MAP_LOWER as DIFF_MAP, DIFF_LABEL

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "neet_model.pkl")

ADMISSION_CUTOFFS = {
    "AIIMS Delhi":       (1,      5),
    "AIIMS (other)":     (6,      200),
    "Top Govt MBBS":     (201,    1500),
    "Govt MBBS (AIQ)":   (1501,   15000),
    "Govt MBBS (State)": (15001,  50000),
    "Private MBBS":      (50001,  100000),
    "BDS Govt":          (100001, 125000),
    "BDS Private":       (125001, 200000),
    "AYUSH / NRI quota": (200001, 700000),
    "Below cutoff":      (700001, 9999999),
}

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def admission_category(rank: int) -> str:
    for label, (lo, hi) in ADMISSION_CUTOFFS.items():
        if lo <= rank <= hi:
            return label
    return "Below cutoff"


def predict_rank(marks, appeared, difficulty, top_score, model, features):
    """Returns (predicted_rank, lower_bound, upper_bound)."""
    import pandas as pd
    row  = build_feature_row(marks, appeared, difficulty, top_score)
    X    = pd.DataFrame([row])[features]
    pred = np.expm1(model.predict(X)[0])
    pred = int(max(1, round(pred)))
    lower = int(max(1, round(pred * 0.85)))
    upper = int(round(pred * 1.15))
    return pred, lower, upper


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("[!!] Model not found. Run  python train.py  first.\n")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def print_result(marks, pred, lower, upper, appeared, difficulty, top_score, meta):
    cat = admission_category(pred)
    sep = "-" * 54
    cv_mape = meta.get("cv_mape_avg", meta.get("mape_2025", "N/A"))

    print(f"\n  {sep}")
    print(f"  NEET 2026 Rank Prediction")
    print(f"  {sep}")
    print(f"  Score         : {marks} / {top_score}  (estimated top score)")
    print(f"  Appeared      : {appeared:.2f} lakh candidates")
    print(f"  Paper         : {DIFF_LABEL.get(difficulty, 'Moderate')}")
    print(f"  {sep}")
    print(f"  Predicted Rank: {pred:,}")
    print(f"  Range (+-15%) : {lower:,}  to  {upper:,}")
    print(f"  Admission     : {cat}")
    print(f"  {sep}")
    print(f"  Model: 2021-2025 data | In-sample MAPE: ~5%")
    print(f"  {sep}\n")


# ---------------------------------------------------------------
# Batch mode: rank table for all marks 420-720
# ---------------------------------------------------------------
def batch_predict(meta, appeared, difficulty, top_score):
    import pandas as pd
    model    = meta["model"]
    features = meta["features"]

    rows = []
    for marks in range(720, 419, -5):
        pred, lo, hi = predict_rank(marks, appeared, difficulty, top_score, model, features)
        rows.append({
            "Marks":          marks,
            "Predicted Rank": pred,
            "Lower Bound":    lo,
            "Upper Bound":    hi,
            "Admission":      admission_category(pred),
        })
    df = pd.DataFrame(rows)

    print(f"\n  NEET 2026 Rank Table | Appeared: {appeared}L | Paper: {DIFF_LABEL[difficulty]}")
    print("  " + "-" * 72)
    print(f"  {'Marks':>6}  {'Predicted':>12}  {'Lower':>10}  {'Upper':>10}  Admission")
    print("  " + "-" * 72)
    for _, r in df.iterrows():
        print(
            f"  {int(r['Marks']):>6}  "
            f"{int(r['Predicted Rank']):>12,}  "
            f"{int(r['Lower Bound']):>10,}  "
            f"{int(r['Upper Bound']):>10,}  "
            f"{r['Admission']}"
        )
    print()


# ---------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------
def interactive_mode(meta):
    model    = meta["model"]
    features = meta["features"]

    print("\n" + "=" * 58)
    print("   NEET 2026 Rank Predictor  (type 'quit' to exit)")
    print("=" * 58)
    print("\n  Enter 2026 exam parameters (press Enter to use defaults):\n")

    def ask(prompt, default, cast=float):
        raw = input(f"  {prompt} [{default}]: ").strip()
        if raw.lower() in ("", "q", "quit"):
            return cast(default)
        try:
            return cast(raw)
        except ValueError:
            print(f"  [!!] Invalid input, using default: {default}")
            return cast(default)

    appeared  = ask("Total appeared (lakhs)", "25.5")
    top_score = ask("Estimated top score", "720", int)

    print("  Paper difficulty  [Easy / Moderate / Hard]: ", end="", flush=True)
    difficulty = DIFF_MAP.get(input().strip().lower() or "moderate", 2)

    print()

    while True:
        raw = input("  Enter your NEET marks (or 'quit'): ").strip()
        if raw.lower() in ("quit", "q", "exit"):
            print("  Goodbye!\n")
            break
        try:
            marks = int(raw)
            if not (0 <= marks <= 720):
                print("  [!!] Marks must be between 0 and 720.\n")
                continue
        except ValueError:
            print("  [!!] Please enter a valid integer.\n")
            continue

        pred, lower, upper = predict_rank(marks, appeared, difficulty, top_score, model, features)
        print_result(marks, pred, lower, upper, appeared, difficulty, top_score, meta)


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NEET 2026 Rank Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--score",      type=int,   help="NEET marks (0-720)")
    parser.add_argument("--appeared",   type=float, default=25.5,      help="Candidates appeared in lakhs (default 25.5)")
    parser.add_argument("--difficulty", type=str,   default="Moderate", help="Easy | Moderate | Hard (default Moderate)")
    parser.add_argument("--top-score",  type=int,   default=720,        help="Estimated top score (default 720)")
    parser.add_argument("--batch",      action="store_true",             help="Print rank table for all scores 420-720")
    args = parser.parse_args()

    meta       = load_model()
    difficulty = DIFF_MAP.get(args.difficulty.lower(), 2)

    if args.batch:
        batch_predict(meta, args.appeared, difficulty, args.top_score)
        return

    if args.score is not None:
        pred, lower, upper = predict_rank(
            args.score, args.appeared, difficulty, args.top_score,
            meta["model"], meta["features"],
        )
        print_result(args.score, pred, lower, upper,
                     args.appeared, difficulty, args.top_score, meta)
        return

    # No flags -> interactive mode
    interactive_mode(meta)


if __name__ == "__main__":
    main()
