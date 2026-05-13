# NEET 2026 Rank Prediction — ML Training Guide

## Dataset Overview
- **File**: `neet_rank_prediction_dataset.csv`
- **Rows**: 253 (historical data points from 2021–2025)
- **Years**: 2021, 2022, 2023, 2024, 2025
- **Mark range**: 414–720 (every 5 marks, with key anchor points)

---

## Column Reference

| Column | Type | Description |
|---|---|---|
| `marks` | int | Raw NEET score (out of 720) |
| `year` | int | Exam year |
| `total_appeared_lakhs` | float | Total candidates appeared (in lakhs) |
| `top_score` | int | Highest score in that year |
| `difficulty_level` | str | Overall paper difficulty: Easy / Moderate / Hard |
| `difficulty_score` | int | Encoded: Easy=1, Moderate=2, Hard=3 |
| `paper_type` | str | Standard or Controversial (2024 grace marks) |
| `physics_difficulty` | str | Section-wise difficulty |
| `chemistry_difficulty` | str | Section-wise difficulty |
| `biology_difficulty` | str | Section-wise difficulty |
| `percentile` | float | Percentile corresponding to marks in that year |
| `air_rank` | int | **TARGET variable** — All India Rank |
| `rank_category` | str | Topper / High / Medium / Low |
| `admission_category` | str | Type of college likely at this rank |

---

## Recommended Features for Training

### Primary Features (high predictive power)
- `marks` — the single strongest predictor
- `total_appeared_lakhs` — more candidates = higher ranks for same score
- `difficulty_score` — paper difficulty shifts rank distribution
- `top_score` — proxy for overall competition ceiling

### Derived Features (engineer these)
```python
df['marks_pct'] = df['marks'] / 720.0          # normalized score
df['score_gap_from_top'] = df['top_score'] - df['marks']  # how far from topper
df['marks_sq'] = df['marks'] ** 2              # non-linear term
df['appeared_x_difficulty'] = df['total_appeared_lakhs'] * df['difficulty_score']
```

### Encoding for categorical columns
```python
diff_map = {'Easy': 1, 'Moderate': 2, 'Hard': 3}
sec_diff_map = {'Easy': 1, 'Moderate': 2, 'Hard': 3}
```

---

## Recommended Models

### 1. Gradient Boosting (best overall)
```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05)
```

### 2. XGBoost (fastest + interpretable)
```python
import xgboost as xgb
model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03)
```

### 3. Random Forest (robust baseline)
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, max_depth=10)
```

### 4. Neural Net (if you have extra features)
```python
# 3-layer MLP: [marks, difficulty, appeared, ...] → rank
# Use log-transform on target (rank) to handle skew
import numpy as np
y_train_log = np.log1p(y_train)
```

---

## Important Notes for 2026 Prediction

### Input assumptions for 2026
```python
input_2026 = {
    'total_appeared_lakhs': 25.0,   # expected ~25-26 lakh
    'difficulty_score': 2,           # assume Moderate (adjust after paper)
    'top_score': 700,                # estimate before result
    'paper_type': 'Standard'
}
```

### Why 2024 data is unreliable
NEET 2024 had 67 candidates score 720/720 due to a grace marks controversy.
The rank distribution is anomalous — consider down-weighting 2024 rows:
```python
df['sample_weight'] = df['year'].map({2021:1.0, 2022:1.0, 2023:1.2, 2024:0.3, 2025:1.5})
```

### Log-transform the target
Air ranks are heavily right-skewed. Always log-transform before training:
```python
import numpy as np
y = np.log1p(df['air_rank'])
# After prediction: rank = np.expm1(model.predict(X))
```

### Evaluation metrics
```python
from sklearn.metrics import mean_absolute_percentage_error
# Target: MAPE < 15% for ranks in 1,000–1,00,000 range
# At very high ranks (1–500): harder due to small sample
```

---

## Train/Test Split Recommendation

Train on 2021–2024, test on 2025:
```python
train = df[df['year'] < 2025]
test  = df[df['year'] == 2025]
```

For cross-validation, use year-based splits (not random), to simulate predicting future years:
```python
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
groups = df['year']
```

---

## Data Sources
- NTA Official NEET Result PDFs (2021–2025)
- Physics Wallah marks vs rank 2025 (official NTA data)
- jeepredictor.in compiled NTA tables
- Careers360 / CollegeDekho verified analysis
- Candidate statistics: NTA press releases

*Note: This dataset is compiled from publicly available official and educational sources. It is intended for educational modelling only. Actual 2026 ranks will be released by NTA on neet.nta.nic.in*