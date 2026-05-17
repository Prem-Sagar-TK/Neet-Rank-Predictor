# -*- coding: utf-8 -*-
"""
app.py - FastAPI backend for NEET 2026 Rank Predictor
======================================================
Run with:
    python app.py
Then open http://localhost:8001 in your browser.
"""

import os
import sys
import sqlite3
import datetime
import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, EmailStr
from typing import Optional

_ROOT      = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.join(_ROOT, "ML")
sys.path.insert(0, SCRIPT_DIR)
from formulas import build_feature_row, DIFF_MAP_LOWER, DIFF_LABEL

# ---------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------
MODEL_PATH = os.path.join(SCRIPT_DIR, "neet_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found. Run  python train.py  first.")

meta     = joblib.load(MODEL_PATH)
MODEL    = meta["model"]
FEATURES = meta["features"]

# ---------------------------------------------------------------
# SQLite database setup
# ---------------------------------------------------------------
DB_PATH = os.path.join(_ROOT, "submissions.db")

def get_db():
    """Return a new SQLite connection (thread-local friendly)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create the submissions table if it doesn't exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                college       TEXT    NOT NULL,
                phone         TEXT    NOT NULL,
                email         TEXT    NOT NULL,
                marks         INTEGER NOT NULL,
                predicted_rank INTEGER,
                category      TEXT,
                submitted_at  TEXT    DEFAULT (datetime('now','localtime'))
            )
        """)
        conn.commit()

init_db()

# ---------------------------------------------------------------
# Admission categories
# ---------------------------------------------------------------
ADMISSION_CUTOFFS = [
    ("AIIMS Delhi",       1,      5,      "#FFD700"),
    ("AIIMS (Other)",     6,      200,    "#C0C0C0"),
    ("Top Govt MBBS",     201,    1500,   "#4ADE80"),
    ("Govt MBBS (AIQ)",   1501,   15000,  "#34D399"),
    ("Govt MBBS (State)", 15001,  50000,  "#60A5FA"),
    ("Private MBBS",      50001,  100000, "#A78BFA"),
    ("BDS Govt",          100001, 125000, "#F472B6"),
    ("BDS Private",       125001, 200000, "#FB7185"),
    ("AYUSH / NRI",       200001, 700000, "#FCD34D"),
    ("Below Cutoff",      700001, 9999999,"#6B7280"),
]

def get_admission(rank: int):
    for label, lo, hi, color in ADMISSION_CUTOFFS:
        if lo <= rank <= hi:
            return label, color
    return "Below Cutoff", "#6B7280"

# ---------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------
class PredictRequest(BaseModel):
    marks:      int   = Field(..., ge=0, le=720)
    appeared:   float = Field(25.5, gt=0)
    difficulty: str   = Field("moderate")
    top_score:  int   = Field(720, ge=600, le=720)

class PredictResponse(BaseModel):
    rank:        int
    lower:       int
    upper:       int
    percentile:  float
    category:    str
    color:       str
    difficulty:  str
    marks:       int
    appeared:    float
    top_score:   int

class BatchItem(BaseModel):
    marks:      int
    rank:       int
    lower:      int
    upper:      int
    category:   str
    color:      str

class SubmitRequest(BaseModel):
    name:           str  = Field(..., min_length=2,  max_length=120)
    college:        str  = Field(..., min_length=2,  max_length=200)
    phone:          str  = Field(..., min_length=7,  max_length=15)
    email:          str  = Field(..., min_length=5,  max_length=200)
    marks:          int  = Field(..., ge=0, le=720)
    predicted_rank: Optional[int]  = None
    category:       Optional[str]  = None

# ---------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------
app = FastAPI(title="NEET 2026 Rank Predictor")

# Serve static files
STATIC_DIR = os.path.join(_ROOT, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

def _predict(marks, appeared, difficulty, top_score):
    diff_enc = DIFF_MAP_LOWER.get(difficulty.lower(), 2)
    row  = build_feature_row(marks, appeared, diff_enc, top_score)
    X    = pd.DataFrame([row])[FEATURES]
    pred = float(np.expm1(MODEL.predict(X)[0]))
    pred = max(1, round(pred))
    lower = max(1, round(pred * 0.85))
    upper = round(pred * 1.15)
    return int(pred), int(lower), int(upper), diff_enc

@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        rank, lower, upper, diff_enc = _predict(
            req.marks, req.appeared, req.difficulty, req.top_score
        )
        total      = req.appeared * 100000
        percentile = round(max(0, (total - rank) / total * 100), 4)
        category, color = get_admission(rank)
        return PredictResponse(
            rank=rank, lower=lower, upper=upper,
            percentile=percentile,
            category=category, color=color,
            difficulty=DIFF_LABEL.get(diff_enc, "Moderate"),
            marks=req.marks, appeared=req.appeared,
            top_score=req.top_score,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/submit")
def submit(req: SubmitRequest):
    """Store a student's details + predicted rank in the database."""
    try:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO submissions
                   (name, college, phone, email, marks, predicted_rank, category)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (req.name, req.college, req.phone, req.email,
                 req.marks, req.predicted_rank, req.category)
            )
            conn.commit()
        return {"status": "ok", "message": "Submission saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/submissions")
def list_submissions(limit: int = 100, offset: int = 0):
    """Return all stored submissions (newest first)."""
    try:
        with get_db() as conn:
            rows = conn.execute(
                """SELECT id, name, college, phone, email, marks,
                          predicted_rank, category, submitted_at
                   FROM submissions
                   ORDER BY id DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset)
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/batch")
def batch(appeared: float = 25.5, difficulty: str = "moderate", top_score: int = 720):
    results = []
    for marks in range(720, 414, -5):
        rank, lower, upper, _ = _predict(marks, appeared, difficulty, top_score)
        category, color = get_admission(rank)
        results.append({
            "marks": marks, "rank": rank,
            "lower": lower, "upper": upper,
            "category": category, "color": color,
        })
    return results

@app.get("/api/health")
def health():
    return {"status": "ok", "model": "neet_2026_ensemble"}

# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n  NEET 2026 Rank Predictor")
    print("  Open -> http://localhost:8001\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
