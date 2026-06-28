"""
data_pipeline.py
----------------
Loads India tech job data (from scraper or CSV), cleans it, stores in SQLite.

Usage:
    python data_pipeline.py --scrape              # scrape fresh data then clean
    python data_pipeline.py --csv data/jobs.csv   # load existing CSV then clean
"""

import sqlite3
import argparse
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

_HERE = Path(__file__).parent

DB_PATH = str(_HERE / "data" / "jobs.db")
TABLE_NAME = "jobs"


# ── Company tier lists ───────────────────────────────────────────────────────
# Classify companies into salary tiers for use as an ML feature.

_FAANG = {
    "google", "alphabet", "meta", "facebook", "instagram", "amazon", "apple",
    "microsoft", "netflix", "uber", "airbnb", "twitter", "x corp", "linkedin",
    "salesforce", "adobe", "nvidia", "intel", "qualcomm", "paypal", "stripe",
    "snowflake", "databricks", "atlassian", "twilio", "cloudflare", "confluent",
    "palantir", "workday", "servicenow", "zendesk", "okta", "datadog",
}

_MNC = {
    "ibm", "oracle", "sap", "accenture", "deloitte", "capgemini", "cognizant",
    "cisco", "dell", "hp", "hewlett", "ericsson", "siemens", "amdocs",
    "dxc", "cgi", "ntt", "fujitsu", "hitachi", "toshiba", "samsung",
    "vmware", "broadcom", "microfocus", "opentext", "teradata", "tibco",
    "bosch", "honeywell", "jp morgan", "jpmorgan", "morgan stanley",
    "goldman sachs", "deutsche bank", "barclays", "mckinsey", "bcg", "bain",
    "kpmg", "pwc", "ernst", "ey ", "thoughtworks", "mastercard", "visa",
    "american express", "amex", "bloomberg", "reuters", "thomson",
    "expedia", "booking", "agoda",
}

_INDIAN_IT = {
    "tcs", "tata consultancy", "infosys", "wipro", "hcl", "hcltech",
    "tech mahindra", "techmahindra", "mphasis", "hexaware", "kpit",
    "birlasoft", "niit", "zensar", "cyient", "l&t infotech",
    "ltimindtree", "lti", "ltts", "persistent", "firstsource", "eclerx",
    "tata ", "infoedge", "naukri", "flipkart", "ola", "paytm", "zomato",
    "swiggy", "razorpay", "freshworks", "zoho", "inmobi", "makemytrip",
}


def _assign_company_tier(company: str) -> str:
    """
    Classify a company name into one of four salary tiers:
      FAANG     → global tech giants (Google, Amazon, Meta, etc.)
      MNC       → large multinationals and consulting firms
      Indian IT → major Indian IT services / product companies
      Startup   → everything else (default)
    """
    if not company or str(company).lower() in ("unknown", "nan", "none", ""):
        return "Startup"
    c = company.lower().strip()
    for name in _FAANG:
        if name in c:
            return "FAANG"
    for name in _MNC:
        if name in c:
            return "MNC"
    for name in _INDIAN_IT:
        if name in c:
            return "Indian IT"
    return "Startup"


# ── Location helpers ─────────────────────────────────────────────────────────

_WFH_RE = re.compile(
    r'^\s*(work\s*(from\s*)?home|remote|wfh|work\s+remotely|anywhere)\s*$',
    re.IGNORECASE,
)


def _normalize_city(raw: str) -> str:
    """
    Extract a clean city name from a raw location string.

    Rules applied in order:
      1. Take the first component before any comma (handles "HSR Layout, Bangalore").
      2. Map WFH / Remote variants to the canonical label "Remote".
    """
    city = str(raw).split(",")[0].strip()
    return "Remote" if _WFH_RE.match(city) else city


def _normalize_date(raw: str) -> str:
    """
    Normalise a date string to ISO format (YYYY-MM-DD).

    Handles:
      "3 days ago"   → today - 3 days
      "2 weeks ago"  → today - 14 days
      "1 month ago"  → today - 30 days
      "2024-07-15"   → "2024-07-15"   (pass-through)
      anything else  → "Not Mentioned"
    """
    s = str(raw).strip()
    today = datetime.now().date()

    m = re.match(r'(\d+)\s+(day|days|week|weeks|month|months)\s+ago', s, re.IGNORECASE)
    if m:
        n, unit = int(m.group(1)), m.group(2).rstrip('s').lower()
        delta = (
            timedelta(days=n)      if unit == "day"   else
            timedelta(weeks=n)     if unit == "week"  else
            timedelta(days=n * 30)                     # month
        )
        return (today - delta).isoformat()

    if re.match(r'\d{4}-\d{2}-\d{2}', s):
        return s[:10]

    return "Not Mentioned"


# ── Column name map ─────────────────────────────────────────────────────────
# Maps many raw header spellings → our clean snake_case names.  Matching is
# case-insensitive and ignores spaces/underscores (see _canonicalize_columns),
# so "Job Title", "job_title", "JOB TITLE" and "jobtitle" all collapse to the
# same key.  This lets arbitrary Kaggle CSVs flow through the same pipeline as
# the scraper output without hand-editing headers per dataset.
COLUMN_MAP = {
    # job_title
    "jobtitle": "job_title", "title": "job_title", "designation": "job_title",
    "role": "job_title", "jobrole": "job_title", "position": "job_title",
    "job": "job_title",
    # company
    "company": "company", "companyname": "company", "employer": "company",
    "organization": "company", "organisation": "company",
    # location
    "location": "location", "city": "location", "joblocation": "location",
    "place": "location",
    # experience (raw string; parsed to experience_years downstream)
    "experience": "experience", "experiencerequired": "experience",
    "yearsofexperience": "experience", "exp": "experience",
    # salary (raw string; parsed to salary_lpa downstream)
    "salary": "salary", "salaryestimate": "salary", "avgsalary": "salary",
    "averagesalary": "salary", "ctc": "salary", "package": "salary",
    "annualsalary": "salary", "salaryinlpa": "salary",
    # skills
    "skills": "skills", "keyskills": "skills", "skill": "skills",
    "skillsrequired": "skills", "techstack": "skills",
    # date_posted
    "dateposted": "date_posted", "date": "date_posted", "posted": "date_posted",
    "postingdate": "date_posted",
}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw CSV headers to our snake_case schema using COLUMN_MAP, matching
    case- and separator-insensitively.  Columns already in canonical form (e.g.
    the scraper's own snake_case output) pass through untouched; unrecognised
    columns are kept as-is.  First match wins when two raw columns map to the
    same target (avoids a downstream column collision).
    """
    rename: dict[str, str] = {}
    taken: set[str] = set()
    for col in df.columns:
        key = re.sub(r"[\s_]+", "", str(col).strip().lower())
        target = COLUMN_MAP.get(key)
        if target and target not in taken and target not in df.columns:
            rename[col] = target
            taken.add(target)
    return df.rename(columns=rename)


def load_csv(path: str) -> pd.DataFrame:
    """Read raw CSV into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    print(f"[✓] Loaded {len(df):,} rows from '{path}'")
    return df


def _parse_experience_years(raw: str) -> float | None:
    """
    Convert experience strings to a single float (midpoint of range).

    Handles:
      "1 to 6 Yrs"    → 3.5
      "3 to 5 Yrs"    → 4.0
      "2 year(s)"     → 2.0
      "5 years"       → 5.0
      "Fresher"       → 0.0
    """
    if "fresher" in raw.lower():
        return 0.0
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", raw)]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 1)


def _parse_salary_lpa(raw: str) -> float | None:
    """
    Convert raw salary strings to LPA (lakhs per annum).

    Handles:
      "₹ 3,00,000 - 5,00,000"  → 4.0   (average of range, rupees → LPA)
      "₹ 2,50,000"             → 2.5
      "₹ 10,00,000"            → 10.0
      "8-12 LPA"               → 10.0  (average, already LPA)
      "₹ 50,000 per month"     → 6.0   (monthly → annualised)
      "Competitive salary"     → None
    """
    # Remove commas so "3,00,000" becomes "300000"
    s = raw.replace(",", "")
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", s)]
    if not nums:
        return None
    avg = sum(nums) / len(nums)
    # Monthly figures must be annualised BEFORE the rupees-vs-LPA heuristic —
    # "₹50,000/month" otherwise parsed as 0.5 LPA and got dropped downstream.
    if re.search(r"per\s*month|/\s*month|monthly|p\.?m\.?\b", raw, re.IGNORECASE):
        avg *= 12
    # If values are in rupees (> 1000), convert to LPA
    if avg > 1000:
        return round(avg / 100_000, 2)
    return round(avg, 2)


def clean(df: pd.DataFrame, source: str = "scrape") -> pd.DataFrame:
    """
    Rename, drop bad rows, parse salary & experience to numbers.

    `source` tags every row (e.g. "scrape", "kaggle:<dataset>") in a `source`
    column so blended training data stays traceable — we can slice metrics by
    origin and tell whether richer external rows actually lift the model.
    """

    # ── Canonicalise headers (case/separator-insensitive) ───────────────────
    df = _canonicalize_columns(df)

    # ── Drop fully empty rows ────────────────────────────────────────────────
    df = df.dropna(how="all")

    # ── Salary ───────────────────────────────────────────────────────────────
    if "salary" in df.columns:
        df["salary_lpa"] = df["salary"].astype(str).map(_parse_salary_lpa)
    else:
        df["salary_lpa"] = None

    # ── Salary band (min/max) — kept alongside the midpoint so the model can
    #    learn from band width and so real vs Adzuna-estimated salaries stay
    #    separable via salary_is_predicted. Missing in pre-band CSVs → None.
    for src, dst in (("salary_min", "salary_min_lpa"),
                     ("salary_max", "salary_max_lpa")):
        if src in df.columns:
            df[dst] = df[src].astype(str).map(_parse_salary_lpa)
        else:
            df[dst] = None

    if "salary_is_predicted" in df.columns:
        df["salary_is_predicted"] = (
            pd.to_numeric(df["salary_is_predicted"], errors="coerce")
              .fillna(0).astype(int)
        )
    else:
        df["salary_is_predicted"] = 0

    # ── Experience ───────────────────────────────────────────────────────────
    if "experience" in df.columns:
        df["experience_years"] = df["experience"].astype(str).map(_parse_experience_years)
    else:
        df["experience_years"] = None

    # ── Normalise city from location ─────────────────────────────────────────
    if "location" in df.columns:
        df["city"] = df["location"].astype(str).map(_normalize_city)
    else:
        df["city"] = "Unknown"

    # ── Normalise date_posted (relative strings → absolute ISO dates) ─────────
    if "date_posted" in df.columns:
        df["date_posted"] = df["date_posted"].astype(str).map(_normalize_date)

    # ── Strip whitespace from text columns ───────────────────────────────────
    for col in ["job_title", "company", "skills"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ── Company tier ─────────────────────────────────────────────────────────
    if "company" in df.columns:
        df["company_tier"] = df["company"].apply(_assign_company_tier)
    else:
        df["company_tier"] = "Startup"

    # ── Impute missing experience from job-title medians ─────────────────────
    # Sources like Indeed don't expose experience on the card; rather than
    # dropping those rows, we fill them with the median years seen for the
    # same normalised title (or the global median as a fallback).
    if "experience_years" in df.columns and bool(df["experience_years"].isna().any()):
        # Build a normalised title key for grouping
        df["_title_key"] = (
            df["job_title"].str.lower()
              .str.replace(r"[^a-z0-9 ]", " ", regex=True)
              .str.split().str[:3].str.join(" ")   # first 3 words
        )
        title_medians   = df.groupby("_title_key")["experience_years"].median()
        global_median   = df["experience_years"].median()
        # Fallback when the whole dataset has no experience data (e.g. Indeed-only run)
        if bool(pd.isna(global_median)):
            global_median = 3.0   # reasonable India tech-job default

        def _fill_exp(row):
            if bool(pd.notna(row["experience_years"])):
                return row["experience_years"]
            med = title_medians.get(row["_title_key"])
            return med if bool(pd.notna(med)) else global_median

        n_missing = df["experience_years"].isna().sum()
        df["experience_years"] = df.apply(_fill_exp, axis=1)
        df.drop(columns=["_title_key"], inplace=True)
        print(f"[✓] Imputed experience_years for {n_missing:,} rows using title-median")

    # ── Drop rows without salary (experience is now always filled) ────────────
    before = len(df)
    df = df.dropna(subset=["salary_lpa"])
    print(f"[✓] Kept {len(df):,} rows after dropping rows without salary (removed {before - len(df):,})")

    # ── Provenance tag — lets us slice metrics by data origin ─────────────────
    df["source"] = source

    df = df.reset_index(drop=True)
    _validate_clean(df)
    return df


def _validate_clean(df: pd.DataFrame) -> None:
    """Sanity-check the cleaned frame before it reaches the DB/model."""
    required = ["job_title", "company", "city", "salary_lpa", "experience_years"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"clean() output is missing required columns: {missing}")
    if len(df) and not pd.api.types.is_numeric_dtype(df["salary_lpa"]):
        raise ValueError("salary_lpa is not numeric after cleaning")
    n_extreme = int(((df["salary_lpa"] < 0.5) | (df["salary_lpa"] > 500)).sum()) if len(df) else 0
    if n_extreme:
        print(f"[!] Warning: {n_extreme} rows have implausible salary_lpa "
              f"(<0.5 or >500) — check _parse_salary_lpa inputs")


# Dedup key — date_posted included so the same role re-posted on a different
# date is kept as a separate listing rather than silently collapsed.
_DEDUP_COLS = ["job_title", "company", "city", "date_posted"]


def save_to_db(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    """
    Merge cleaned DataFrame into SQLite, preserving historical rows.

    Incremental: deletes existing rows that share a dedup key with incoming
    rows (newest wins), then appends — instead of reading the whole table
    into pandas and rewriting it, which scaled O(n) with DB size.
    Falls back to a one-time full merge when the schema has changed.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        existing_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({TABLE_NAME})")]

        # ── Fresh database ────────────────────────────────────────────────────
        if not existing_cols:
            df = df.drop_duplicates(subset=[c for c in _DEDUP_COLS if c in df.columns],
                                    keep="last")
            df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
            print(f"[✓] +{len(df):,} rows → '{db_path}' (new table: {TABLE_NAME})")
            return

        # ── Schema changed (e.g. new column) → one-time full merge ────────────
        if set(existing_cols) != set(df.columns):
            existing = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
            combined = pd.concat([existing, df], ignore_index=True)
            # Backfill company_tier for rows that predate the column
            if "company_tier" not in existing.columns and "company" in combined.columns:
                combined["company_tier"] = combined["company"].apply(_assign_company_tier)
            # Legacy rows predate the provenance tag → they were all scraped.
            if "source" in combined.columns:
                combined["source"] = combined["source"].fillna("scrape")
            dedup_cols = [c for c in _DEDUP_COLS if c in combined.columns]
            if dedup_cols:
                combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
            combined.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
            print(f"[✓] Schema migration: +{len(combined) - len(existing):,} rows → "
                  f"'{db_path}' (total: {len(combined):,})")
            return

        # ── Normal incremental path ───────────────────────────────────────────
        dedup_cols = [c for c in _DEDUP_COLS if c in df.columns]
        df = df.drop_duplicates(subset=dedup_cols, keep="last")

        where = " AND ".join(f"{c} = ?" for c in dedup_cols)
        keys  = [tuple(str(row[c]) for c in dedup_cols) for _, row in df[dedup_cols].iterrows()]
        replaced = 0
        cur = conn.cursor()
        for key in keys:
            replaced += cur.execute(
                f"DELETE FROM {TABLE_NAME} WHERE {where}", key
            ).rowcount

        df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        total = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]

    print(f"[✓] +{len(df) - replaced:,} new rows ({replaced:,} updated) → "
          f"'{db_path}' (total: {total:,}, table: {TABLE_NAME})")


def load_from_db(db_path: str = DB_PATH) -> pd.DataFrame:
    """Read the cleaned jobs table back from SQLite."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    return df


def run(csv_path: str, source: str = "scrape") -> pd.DataFrame:
    df_raw = load_csv(csv_path)
    df_clean = clean(df_raw, source=source)
    save_to_db(df_clean)
    return df_clean


def run_with_scraper(pages: int = 3, keywords: list | None = None) -> pd.DataFrame:
    """Scrape fresh data, then clean and store it."""
    from scraper import scrape
    csv_path = scrape(keywords=keywords, pages_per_keyword=pages)
    if not os.path.exists(csv_path):
        raise RuntimeError(
            "Scraping failed — no CSV was created.\n"
            "  • Check your internet connection\n"
            "  • Check your API credentials in .env and try again in a few minutes"
        )
    return run(csv_path, source="scrape")


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load & clean India jobs data into SQLite")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv",    help="Path to existing CSV file")
    group.add_argument("--scrape", action="store_true", help="Fetch fresh data via API (Adzuna / JSearch)")
    parser.add_argument("--pages", type=int, default=3, help="Pages per keyword when scraping (default: 3)")
    parser.add_argument("--source", default=None,
                        help="Provenance tag for --csv rows (e.g. 'kaggle:india-tech-salaries'). "
                             "Defaults to 'scrape'.")
    args = parser.parse_args()

    if args.scrape:
        run_with_scraper(pages=args.pages)
    else:
        run(args.csv, source=args.source or "scrape")
