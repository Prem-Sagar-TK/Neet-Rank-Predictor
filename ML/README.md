# NEET 2026 Rank Predictor

A machine-learning ensemble that predicts All-India Ranks (AIR) for NEET 2026 based on a student's score and exam-year conditions.

---

## Project Structure

```
Neet Rank Predictor/
├── neet_rank_prediction_dataset.csv   # Historical data 2021-2025 (253 rows)
├── model.py                           # Feature schema + ensemble factory (shared)
├── formulas.py                        # Feature-engineering reference
├── train.py                           # Training pipeline -> saves neet_model.pkl
├── predict.py                         # CLI predictor
├── requirements.txt
└── claude.md                          # Dataset documentation
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step 1 — Train the model

```bash
python train.py
```

This will:
- Load `neet_rank_prediction_dataset.csv`
- Engineer 15 features (marks, difficulty, appeared, interaction terms)
- Down-weight 2024 data (grace-marks controversy)
- Run Leave-One-Year-Out cross-validation across all 5 years
- Train a final **VotingRegressor ensemble** (GradientBoosting + RandomForest + XGBoost)
- Save the fitted model to `neet_model.pkl`

---

## Step 2 — Predict ranks

### Interactive mode (prompts for all inputs)

```bash
python predict.py
```

### Single score (command-line flags)

```bash
python predict.py --score 600
python predict.py --score 580 --difficulty Hard --appeared 25.0 --top-score 690
```

### Full rank table for all scores 420–720

```bash
python predict.py --batch
python predict.py --batch --difficulty Hard --appeared 25.5 --top-score 690
```

---

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--score` | (none) | Your NEET marks (0–720) |
| `--appeared` | `25.5` | Total candidates appeared (in lakhs) |
| `--difficulty` | `Moderate` | Overall paper difficulty: Easy / Moderate / Hard |
| `--top-score` | `720` | Estimated top score for the year |
| `--phys` | `Moderate` | Physics section difficulty |
| `--chem` | `Moderate` | Chemistry section difficulty |
| `--bio` | `Moderate` | Biology section difficulty |
| `--batch` | (flag) | Print rank table for every 5-mark interval |

---

## Model details

| Component | Algorithm | Hyperparameters |
|---|---|---|
| GB | GradientBoostingRegressor | n=300, depth=5, lr=0.05, subsample=0.85 |
| RF | RandomForestRegressor | n=200, depth=10 |
| XGB | XGBRegressor | n=500, depth=6, lr=0.03 |
| Ensemble | VotingRegressor (equal weights) | — |

**Target transformation**: `log1p(air_rank)` — handles the heavy right-skew in rank distribution.

**In-sample fit** (all 253 rows): MAPE ~5%, R² ~0.999

---

## 2026 Exam Assumptions (adjustable after paper release)

| Parameter | Default | Notes |
|---|---|---|
| Total appeared | 25.5 lakh | ~25–26 lakh expected |
| Overall difficulty | Moderate | Adjust after May 4, 2026 |
| Top score | 720 | Adjust after result |

---

## Important caveats

- **Sample size is small**: Only 5 exam years (2021–2025). Year-to-year LOO cross-validation errors are high (50–80% MAPE for normal years) because each year's rank distribution depends heavily on candidate pool, paper difficulty, and NTA decisions — factors not fully captured with 5 data points.
- **2024 data is anomalous**: 67 candidates scored 720/720 due to grace marks. That year is down-weighted (0.3x) in training.
- **2025 is hard paper**: Top score was 686, which was unprecedented. The model has now learned from this year.
- **This is an educational tool**: Actual 2026 ranks will be released by NTA on [neet.nta.nic.in](https://neet.nta.nic.in).
