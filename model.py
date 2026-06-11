"""
model.py  —  Phase 1 upgrade
-----------------------------
Improvements over v1:
  • Multi-hot encodes top-20 skills (each skill → 0/1 column)
  • Adds company as an encoded feature
  • Trains both Random Forest AND XGBoost, picks the winner
  • Evaluates with 5-fold cross-validation (more reliable than single split)
  • Saves feature importance to models/feature_importance.json for the dashboard

Usage:
    python model.py            # trains, compares, saves best model
"""

import difflib
import json
import os
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from data_pipeline import load_from_db, _assign_company_tier

warnings.filterwarnings("ignore")

LOG_TRANSFORM_TARGET = True

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE            = Path(__file__).parent
MODEL_PATH       = str(_HERE / "models" / "salary_model.pkl")
ENCODERS_PATH    = str(_HERE / "models" / "encoders.pkl")
IMPORTANCE_PATH  = str(_HERE / "models" / "feature_importance.json")
METRICS_PATH     = str(_HERE / "models" / "model_metrics.json")
TOP_SKILLS_PATH  = str(_HERE / "models" / "top_skills.pkl")

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_COL          = "salary_lpa"
N_SKILLS            = 20     # how many top skills to multi-hot encode
CV_FOLDS            = 5      # k-fold cross-validation
MIN_ENCODER_SAMPLES = 3      # groups with fewer training rows → use global_mean
FUZZY_CUTOFF        = 0.85   # strict: only accept near-identical string matches

# ── City aliases ──────────────────────────────────────────────────────────────
# Maps satellite cities, suburbs, and alternate spellings to one of the 6
# INDIA_CITIES that the scraper targets.  Applied before target encoding so the
# model sees "Delhi" instead of "Gurugram", etc.
_CITY_ALIASES: dict[str, str] = {
    # Bangalore
    "bengaluru": "Bangalore", "bengalure": "Bangalore",
    # Delhi NCR
    "new delhi": "Delhi", "delhi ncr": "Delhi",
    "noida": "Delhi", "greater noida": "Delhi", "greater noida west": "Delhi",
    "gurugram": "Delhi", "gurgaon": "Delhi",
    "faridabad": "Delhi", "ghaziabad": "Delhi",
    # Mumbai Metropolitan Region
    "navi mumbai": "Mumbai", "thane": "Mumbai",
    "andheri": "Mumbai", "andheri east": "Mumbai",
    "bandra": "Mumbai", "powai": "Mumbai", "malad": "Mumbai",
    "borivali": "Mumbai", "goregaon": "Mumbai",
    "vikhroli": "Mumbai", "lower parel": "Mumbai", "parel": "Mumbai",
    "bkc": "Mumbai", "airoli": "Mumbai", "tardeo": "Mumbai",
    # Chennai
    "adyar": "Chennai", "anna nagar": "Chennai", "t nagar": "Chennai",
    "velachery": "Chennai", "pallikaranai": "Chennai",
    "keelkattalai": "Chennai", "teynampet": "Chennai",
    # Hyderabad
    "secunderabad": "Hyderabad", "madhapur": "Hyderabad",
    "begumpet": "Hyderabad", "hitec city": "Hyderabad",
    # Pune
    "pimpri": "Pune", "hinjewadi": "Pune", "hadapsar": "Pune",
    "wakad": "Pune", "kharadi": "Pune",
    # Bangalore sub-areas
    "banaswadi": "Bangalore", "whitefield": "Bangalore",
    "hsr layout": "Bangalore", "koramangala": "Bangalore",
    "marathahalli": "Bangalore", "electronic city": "Bangalore",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_top_skills(df: pd.DataFrame, n: int = N_SKILLS) -> list[str]:
    """Return the n most frequent skills across all job listings."""
    counts = {}
    for row in df["skills"].dropna():
        for skill in str(row).split(","):
            s = skill.strip().lower()
            if s and s != "not mentioned" and len(s) > 1:
                counts[s] = counts.get(s, 0) + 1
    top = sorted(counts, key=counts.get, reverse=True)[:n]
    return top


def _multihot_skills(df: pd.DataFrame, top_skills: list[str]) -> pd.DataFrame:
    """
    Add one binary column per top skill.
    skill_python = 1 if 'python' appears in that job's skills string, else 0.
    """
    skills_str = df["skills"].fillna("").astype(str).str.lower()
    for skill in top_skills:
        col = f"skill_{skill.replace(' ', '_')}"
        df[col] = skills_str.str.contains(skill, regex=False).astype(int)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def _extract_seniority(title: str) -> int:
    """
    Extract a 0–4 seniority ordinal from the raw job title.

    MUST be called before _normalize_title, which strips the seniority words.
    Seniority and role type are orthogonal signals: "Senior Data Scientist" and
    "Data Scientist Intern" both normalise to the same title bucket but have
    very different salaries.  Encoding them separately lets the model capture
    both dimensions without fragmenting the title encoder into hundreds of
    low-count groups.

    Levels:
      0 — intern / fresher / trainee
      1 — junior / associate
      2 — mid-level (default, no explicit marker)
      3 — senior / lead / principal
      4 — head / VP / director / chief
    """
    # Pad with spaces so keywords match whole words without regex overhead.
    t = " " + str(title).lower() + " "
    if any(kw in t for kw in [" head of ", " vp ", " director ", " chief "]):
        return 4
    if any(kw in t for kw in [" principal ", " lead ", " senior ", " sr. ", " sr "]):
        return 3
    if any(kw in t for kw in [" junior ", " jr. ", " jr ", " associate "]):
        return 1
    if any(kw in t for kw in [" intern ", " internship ", " fresher ", " trainee "]):
        return 0
    return 2


# Seniority/noise words stripped from titles before bucketing.
# Longest-first so "internship" wins over "intern"; lookarounds (not \b, which
# misbehaves around "sr.") keep whole words intact — plain str.replace mangled
# titles ("International" → "ational", "Team Leader" → "Team er").
_TITLE_NOISE_RE = re.compile(
    r"(?<!\w)(" + "|".join(sorted(map(re.escape, [
        "senior", "sr.", "sr", "junior", "jr.", "jr", "lead", "principal",
        "associate", "internship", "intern", "fresher", "trainee",
        "executive", "manager", "head of", "vp of",
    ]), key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)


def _normalize_title(title: str) -> str:
    """
    Collapse noisy job title variants into ~12 canonical buckets.

    The catch-all "Other Tech" prevents hundreds of rare titles from becoming
    unique target-encoder categories — groups with a single sample produce
    meaningless LOO estimates and unseen labels hurt test-set predictions.
    """
    t = str(title).lower().strip()
    # Strip seniority prefixes/suffixes as WHOLE WORDS only — plain str.replace
    # mangled titles ("International" → "ational", "Team Leader" → "Team er").
    t = _TITLE_NOISE_RE.sub(" ", t)
    t = " ".join(t.split())  # collapse whitespace

    # Map to canonical titles — order matters (first match wins)
    mapping = [
        (["data scientist", "data science"],                    "Data Scientist"),
        (["machine learning", "ml engineer", "ai engineer",
          "deep learning", "nlp", "artificial intelligence",
          "generative ai", "gen ai", "llm", "prompt engineer",
          "ai developer", "ai architect"],                      "ML Engineer"),
        (["data analyst", "business analyst", "analytics",
          "bi analyst", "research analyst"],                    "Data Analyst"),
        (["data engineer", "etl", "pipeline",
          "databricks", "spark", "pyspark",
          "data ops", "data modell"],                           "Data Engineer"),
        (["database", "sql developer", "dba", "pl/sql",
          "oracle developer", "mysql", "postgresql"],           "Database Engineer"),
        (["qa", "quality assurance", "test engineer",
          "tester", "automation test", "sdet", "testing"],      "QA Engineer"),
        (["devops", "cloud engineer", "site reliability",
          "sre", "platform engineer",
          "kubernetes", "terraform", "infrastructure"],         "DevOps Engineer"),
        (["frontend", "front end", "react", "angular",
          "ui developer", "web developer", "vue"],              "Frontend Engineer"),
        (["python developer", "python engineer",
          "backend", "software engineer", "sde",
          "software developer", "full stack", "fullstack",
          "java developer", "node", "django",
          ".net", "spring boot", "mern", "mean stack"],         "Software Engineer"),
        (["product manager", "product owner"],                  "Product Manager"),
        (["bi developer", "business intelligence",
          "tableau", "power bi", "looker", "qlik"],             "BI Developer"),
        (["security", "cyber", "ethical hacker",
          "penetration", "soc analyst", "information security"], "Security Engineer"),
    ]
    for keywords, canonical in mapping:
        if any(kw in t for kw in keywords):
            return canonical
    # Catch-all: avoids creating hundreds of one-off encoder categories for
    # niche titles like "Garment Production Manager", "Tinkering Coordinator" etc.
    return "Other Tech"


def _build_target_encoders(df: pd.DataFrame) -> dict:
    """
    Group-mean target encoding stats: per-category sum and count of salary_lpa,
    plus the global mean.  Categories with < MIN_ENCODER_SAMPLES rows fall back
    to the global mean at lookup time (see _apply_target_encoding).
    """
    global_mean = df[TARGET_COL].mean()
    encoders = {"global_mean": global_mean}
    for col in ["job_title", "city", "company"]:
        group = df.groupby(col)[TARGET_COL].agg(["sum", "count"])
        encoders[col] = group
    return encoders


def _apply_target_encoding(df: pd.DataFrame, col: str,
                            encoders: dict) -> pd.Series:
    """
    Both training and inference use the same group-mean encoding so that
    training and test features occupy the same value space.

    Why not LOO: LOO subtracts each row's own salary, making every training row's
    encoded value unique (~1/N variation per row).  The RF memorises these N unique
    values → CV looks perfect, test R² collapses.  Group mean gives 12 canonical
    values for 12 title buckets — consistent across train and test, and the RF
    actually learns to generalise.

    Residual leakage: each row contributes 1/N to its own group mean (negligible
    for large groups; groups < MIN_ENCODER_SAMPLES fall back to global_mean).
    """
    global_mean = encoders["global_mean"]
    group = encoders[col]   # DataFrame with sum/count per category
    known = group.index.tolist()

    def lookup(cat):
        if cat in group.index:
            s, c = group.loc[cat, "sum"], group.loc[cat, "count"]
            if c < MIN_ENCODER_SAMPLES:
                return global_mean
            return s / c
        # Strict fuzzy fallback — FUZZY_CUTOFF=0.85 prevents cross-domain
        # mismatches like 'ML Researcher' → 'Teacher' (cutoff=0.4 caused these).
        # Anything below the threshold falls back to global_mean.
        matches = difflib.get_close_matches(cat, known, n=1, cutoff=FUZZY_CUTOFF)
        if matches:
            return group.loc[matches[0], "sum"] / group.loc[matches[0], "count"]
        return global_mean

    # Resolve each UNIQUE category once, then broadcast — the fuzzy fallback is
    # O(categories) per lookup, so per-row mapping was O(rows × categories).
    col_str = df[col].astype(str)
    mapping = {cat: lookup(cat) for cat in col_str.unique()}
    return col_str.map(mapping)


def build_features(df: pd.DataFrame, encoders: dict = None,
                   top_skills: list = None, fit: bool = True):
    """
    Full feature engineering pipeline.
    Returns (X DataFrame, encoders dict, top_skills list, feature_cols list)
    """
    df = df.copy()

    # ── Seniority — extract BEFORE normalisation strips the words ─────────────
    # "Senior Data Scientist" and "Data Analyst Intern" normalise to the same
    # title bucket; seniority_enc captures the salary signal those words carry.
    df["seniority_enc"] = df["job_title"].fillna("").map(_extract_seniority)

    # ── Normalize job titles ──────────────────────────────────────────────────
    df["job_title"] = df["job_title"].fillna("Unknown").apply(_normalize_title)

    # ── Normalize cities — map aliases/suburbs to canonical INDIA_CITIES ─────
    def _resolve_city(raw: str) -> str:
        c = str(raw).strip()
        return _CITY_ALIASES.get(c.lower(), c.title())
    df["city"]  = df["city"].fillna("Unknown").map(_resolve_city)

    df["company"] = df["company"].fillna("Unknown").astype(str).str.strip()

    # ── Target encode: job_title, city, company ───────────────────────────────
    # Each column becomes its mean salary — far more informative than a random int.
    if fit:
        encoders = _build_target_encoders(df)

    for col in ["job_title", "city", "company"]:
        df[f"{col}_enc"] = _apply_target_encoding(df, col, encoders)

    # ── Multi-hot encode skills ───────────────────────────────────────────────
    if fit:
        top_skills = _extract_top_skills(df, N_SKILLS)
    df = _multihot_skills(df, top_skills)

    # ── Company tier (ordinal) ────────────────────────────────────────────────
    # FAANG pays most → highest ordinal. Startup is the default/baseline.
    _TIER_ORDER = {"Startup": 0, "Indian IT": 1, "MNC": 2, "FAANG": 3}
    if "company_tier" not in df.columns:
        # Derive tier from company column (covers predict() path and old DB rows)
        df["company_tier"] = df["company"].fillna("Unknown").apply(_assign_company_tier)
    df["company_tier_enc"] = df["company_tier"].map(lambda t: _TIER_ORDER.get(t, 0))

    # ── Final feature columns ─────────────────────────────────────────────────
    skill_cols   = [f"skill_{s.replace(' ', '_')}" for s in top_skills]
    feature_cols = (["job_title_enc", "city_enc", "company_enc",
                     "company_tier_enc", "seniority_enc", "experience_years"] + skill_cols)
    feature_cols = [c for c in feature_cols if c in df.columns]

    return df[feature_cols], encoders, top_skills, feature_cols


# ── Training ──────────────────────────────────────────────────────────────────

def _clean_salary(df: pd.DataFrame,
                  high_fence: float | None = None) -> tuple[pd.DataFrame, float]:
    """
    Two-pass salary cleaning.  Returns (cleaned_df, high_fence).

    Pass high_fence to reuse bounds computed from another split — e.g. fit on
    df_train, then pass the same fence when filtering df_test so that test-set
    salary statistics do not influence the preprocessing step.

      1. Hard floor at 1.0 LPA — removes zeros and bad parse artefacts.
      2. IQR upper fence — computed from THIS df if high_fence is None,
         otherwise the supplied value is used directly.
    """
    before = len(df)

    # Pass 1: hard floor
    df = df[df[TARGET_COL] >= 1.0]
    removed_bad = before - len(df)
    if removed_bad:
        print(f"[✓] Dropped {removed_bad} rows with salary < ₹1 LPA (bad parse)")

    # Pass 2: IQR upper fence
    if high_fence is None:
        q1  = df[TARGET_COL].quantile(0.25)
        q3  = df[TARGET_COL].quantile(0.75)
        iqr = q3 - q1
        high_fence = q3 + 1.5 * iqr
    before2 = len(df)
    df = df[df[TARGET_COL] <= high_fence]
    removed_high = before2 - len(df)
    if removed_high:
        print(f"[✓] Removed {removed_high} high outliers  (capped at ₹{high_fence:.1f} LPA)")

    sal = df[TARGET_COL]
    print(f"[✓] Clean salary range: ₹{sal.min():.1f}–{sal.max():.1f} LPA  "
          f"(median ₹{sal.median():.1f} LPA)")
    return df.reset_index(drop=True), high_fence


def train() -> tuple:
    """
    Load data, engineer features, train RF + XGBoost,
    compare with 5-fold CV, save the winner.
    Returns (best_model, encoders, top_skills, feature_cols, metrics_dict)
    """
    print("[→] Loading data from database...")
    df = load_from_db()

    # Drop rows without salary
    df = df.dropna(subset=[TARGET_COL, "experience_years"])
    print(f"[✓] {len(df):,} samples with salary + experience")

    # ── Split FIRST — before any statistics are computed on the data ─────────
    # 1. Splitting before _clean_salary means the IQR fence is fitted on
    #    df_train only; test salary distribution no longer influences which
    #    rows are treated as outliers.
    # 2. build_features() encodes categories with training group means, so
    #    splitting first also prevents target-encoding leakage.
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    # ── Clean salary — fit bounds on train, apply same bounds to test ─────────
    df_train, high_fence = _clean_salary(df_train)
    df_test,  _          = _clean_salary(df_test, high_fence=high_fence)

    # ── Feature engineering (fit on train, apply to test) ─────────────────────
    X_train, encoders, top_skills, feature_cols = build_features(df_train, fit=True)
    X_test,  _,        _,          _            = build_features(
        df_test, encoders=encoders, top_skills=top_skills, fit=False
    )

    # ── Target: log-transform if enabled ─────────────────────────────────────
    y_raw_train = df_train[TARGET_COL].values
    y_raw_test  = df_test[TARGET_COL].values
    if LOG_TRANSFORM_TARGET:
        y_train = np.log1p(y_raw_train)
        y_test  = np.log1p(y_raw_test)
        print(f"[✓] Log-transformed target (log1p).  "
              f"Salary range: ₹{y_raw_train.min():.1f}–{y_raw_train.max():.1f} LPA (train)")
    else:
        y_train = y_raw_train
        y_test  = y_raw_test

    print(f"[✓] Feature matrix: {X_train.shape[0]} train + {X_test.shape[0]} test  ×  {X_train.shape[1]} features")

    # ── Define models ─────────────────────────────────────────────────────────
    models = {
        # max_depth capped — unbounded trees memorise the small dataset and
        # inflate the train/test gap; min_samples_leaf alone wasn't enough.
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0
        ),
    }

    # ── 5-fold cross-validation on train set ─────────────────────────────────
    # n_jobs=1 here avoids thread contention with RF's own n_jobs=-1 parallelism.
    print(f"\n[→] Running {CV_FOLDS}-fold cross-validation on train set...")
    cv_results = {}
    for name, m in models.items():
        scores = cross_val_score(
            m, X_train, y_train,
            cv=CV_FOLDS,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        mae_scores = -scores   # log-space MAE (unitless if log-transformed)
        cv_results[name] = {
            "cv_mae_log":  round(float(mae_scores.mean()), 4),
            "cv_mae_std":  round(float(mae_scores.std()),  4),
        }
        if LOG_TRANSFORM_TARGET:
            print(f"  {name:20s} → CV MAE (log): {mae_scores.mean():.4f} ± {mae_scores.std():.4f}  "
                  f"[≈ ×{np.exp(mae_scores.mean()):.2f} multiplicative error]")
        else:
            print(f"  {name:20s} → CV MAE: {mae_scores.mean():.3f} ± {mae_scores.std():.3f} LPA")

    # ── Pick winner (lowest CV MAE) ───────────────────────────────────────────
    best_name = min(cv_results, key=lambda k: cv_results[k]["cv_mae_log"])
    print(f"\n[✓] Winner: {best_name}")

    # ── Fit winner on full train set; evaluate on held-out test set ───────────
    best_model = models[best_name]
    best_model.fit(X_train.values, y_train)
    y_pred = best_model.predict(X_test.values)

    # Back-transform predictions to LPA for human-readable metrics
    if LOG_TRANSFORM_TARGET:
        y_pred_lpa  = np.expm1(y_pred)
        y_test_lpa  = np.expm1(y_test)
    else:
        y_pred_lpa  = y_pred
        y_test_lpa  = y_test

    mae    = mean_absolute_error(y_test_lpa, y_pred_lpa)
    r2     = r2_score(y_test,     y_pred)         # R² on log scale
    r2_lpa = r2_score(y_test_lpa, y_pred_lpa)     # R² on raw LPA scale
    print(f"[✓] Test MAE : ₹{mae:.2f} LPA")
    print(f"[✓] R² (log) : {r2:.3f}  |  R² (LPA) : {r2_lpa:.3f}")

    # ── Feature importance ────────────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        importance_dict = dict(zip(feature_cols, [round(float(i), 4) for i in importances]))
        importance_sorted = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    else:
        importance_sorted = {}

    # ── Full metrics ──────────────────────────────────────────────────────────
    metrics = {
        "best_model":        best_name,
        "test_mae_lpa":      round(mae, 2),
        "test_r2":           round(r2_lpa, 3),
        "log_transform":     LOG_TRANSFORM_TARGET,
        "n_features":        len(feature_cols),
        "n_samples":         len(df_train) + len(df_test),
        "cv_results":        cv_results,
        "feature_importance": importance_sorted,
    }

    return best_model, encoders, top_skills, feature_cols, metrics


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model, encoders, top_skills, feature_cols, metrics) -> None:
    # Absolute path — "models" relative to cwd silently diverged from the
    # absolute MODEL_PATH/... constants when run from another directory.
    os.makedirs(_HERE / "models", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols,
                     "log_transform": LOG_TRANSFORM_TARGET}, f)

    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    with open(TOP_SKILLS_PATH, "wb") as f:
        pickle.dump(top_skills, f)

    with open(IMPORTANCE_PATH, "w") as f:
        json.dump(metrics["feature_importance"], f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump({
            "best_model": metrics["best_model"],
            "test_mae_lpa": metrics["test_mae_lpa"],
            "test_r2": metrics["test_r2"],
            "n_samples": metrics["n_samples"],
            "n_features": metrics["n_features"],
        }, f, indent=2)

    print(f"[✓] Model saved        → {MODEL_PATH}")
    print(f"[✓] Encoders saved     → {ENCODERS_PATH}")
    print(f"[✓] Feature importance → {IMPORTANCE_PATH}")
    print(f"[✓] Model metrics      → {METRICS_PATH}")


_MODEL_CACHE: dict = {}


def load_model(force_reload: bool = False):
    """Load model artefacts from disk, cached after the first call.

    The cache is invalidated automatically when salary_model.pkl's mtime
    changes (e.g. after a retrain), or explicitly via force_reload=True.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run: python model.py")

    mtime = os.path.getmtime(MODEL_PATH)
    if not force_reload and _MODEL_CACHE.get("mtime") == mtime:
        return _MODEL_CACHE["artefacts"]

    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    with open(TOP_SKILLS_PATH, "rb") as f:
        top_skills = pickle.load(f)

    artefacts = (payload["model"], encoders, top_skills, payload["feature_cols"],
                 payload.get("log_transform", LOG_TRANSFORM_TARGET))
    _MODEL_CACHE.update({"mtime": mtime, "artefacts": artefacts})
    return artefacts


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(job_title: str, city: str, experience_years: float,
            skills: str = "", company: str = "Unknown") -> dict:
    """
    Predict salary.
    Args:
        job_title:        e.g. "Data Analyst"
        city:             e.g. "Bangalore"
        experience_years: e.g. 3.0
        skills:           comma-separated, e.g. "Python, SQL, Machine Learning"
        company:          optional, e.g. "Infosys"
    """
    model, encoders, top_skills, feature_cols, log_transform = load_model()

    input_df = pd.DataFrame([{
        "job_title":        _normalize_title(job_title),
        "city":             city,
        "company":          company,
        "experience_years": float(experience_years),
        "skills":           skills,
    }])

    X, _, _, _ = build_features(input_df, encoders=encoders,
                                 top_skills=top_skills, fit=False)

    # Align columns to training order (fill missing with 0)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    # Confidence interval via individual tree predictions (RF only; XGBoost falls back to ±15%)
    try:
        trees = model.estimators_   # AttributeError for XGBRegressor → except branch
        tree_preds = np.array([t.predict(X.values)[0] for t in trees])
        if log_transform:
            tree_preds = np.expm1(tree_preds)
        predicted = round(float(np.mean(tree_preds)), 2)
        low       = round(float(np.percentile(tree_preds, 10)), 2)
        high      = round(float(np.percentile(tree_preds, 90)), 2)
    except AttributeError:
        raw_pred = float(model.predict(X.values)[0])
        if log_transform:
            raw_pred = float(np.expm1(raw_pred))
        predicted = round(raw_pred, 2)
        low, high = round(predicted * 0.85, 2), round(predicted * 1.15, 2)

    return {
        "job_title":             job_title,
        "city":                  city,
        "experience_years":      experience_years,
        "skills":                skills or "Not specified",
        "predicted_salary_lpa":  predicted,
        "range_low_lpa":         low,
        "range_high_lpa":        high,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, encoders, top_skills, feature_cols, metrics = train()
    save_model(model, encoders, top_skills, feature_cols, metrics)

    print("\n[── Model Metrics ──]")
    print(f"  Best model  : {metrics['best_model']}")
    print(f"  Test MAE    : ₹{metrics['test_mae_lpa']} LPA")
    print(f"  R² Score    : {metrics['test_r2']}")
    print(f"  Features    : {metrics['n_features']}")
    print(f"  Samples     : {metrics['n_samples']}")

    print("\n[── CV Results ──]")
    for name, res in metrics["cv_results"].items():
        print(f"  {name:20s} → log MAE: {res['cv_mae_log']} ± {res['cv_mae_std']}")

    print("\n[── Top 10 Important Features ──]")
    for feat, imp in list(metrics["feature_importance"].items())[:10]:
        bar = "█" * int(imp * 200)
        print(f"  {feat:35s} {imp:.4f}  {bar}")
