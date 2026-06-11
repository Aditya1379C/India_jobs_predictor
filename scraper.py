"""
scraper.py
----------
Fetches India tech job listings via API (no HTML scraping).

Supported sources (configured via .env):
  • Adzuna   — https://developer.adzuna.com  (free: 250 calls/day)
  • JSearch  — https://rapidapi.com/letscrape-6bfp2kogce/api/jsearch
               (free: 500 calls/month, aggregates Indeed / LinkedIn / Glassdoor)

Setup:
  1. Copy .env.example → .env and fill in your key(s).
  2. Set API_SOURCE=adzuna  OR  API_SOURCE=jsearch  in .env.
  3. Run:  python scraper.py

Usage:
    python scraper.py                        # default keywords, all cities
    python scraper.py --pages 3              # pages per keyword (Adzuna only)
    python scraper.py --source adzuna        # force a specific API
    python scraper.py --source jsearch
"""

import argparse
import csv
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent

API_SOURCE = os.getenv("API_SOURCE", "adzuna").lower()   # "adzuna" | "jsearch"

# Adzuna marks many Indian salaries as model-estimated (salary_is_predicted).
# Set INCLUDE_PREDICTED_SALARIES=false in .env to treat those as "Not Mentioned"
# (cleaner training target, but far fewer salary samples).
INCLUDE_PREDICTED_SALARIES = (
    os.getenv("INCLUDE_PREDICTED_SALARIES", "true").lower() != "false"
)

# Adzuna
ADZUNA_APP_ID  = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")
ADZUNA_BASE    = "https://api.adzuna.com/v1/api/jobs/in/search"

# JSearch (RapidAPI)
JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY", "")
JSEARCH_BASE    = "https://jsearch.p.rapidapi.com/search"

OUTPUT_PATH = str(_HERE / "data" / "scraped_jobs.csv")

CSV_HEADERS = [
    "job_title", "company", "location", "experience",
    "salary", "skills", "date_posted",
]

# ── Keywords & cities ─────────────────────────────────────────────────────────

DEFAULT_KEYWORDS = [
    "data analyst",
    "data scientist",
    "data engineer",
    "business analyst",
    "business intelligence",
    "analytics engineer",
    "machine learning engineer",
    "AI engineer",
    "MLOps engineer",
    "NLP engineer",
    "python developer",
    "software engineer",
    "backend developer",
    "full stack developer",
    "cloud engineer",
    "devops engineer",
    "SQL developer",
    "ETL developer",
]

INDIA_CITIES = [
    "Bangalore",
    "Mumbai",
    "Hyderabad",
    "Delhi",
    "Pune",
    "Chennai",
]

# Tech skills to detect in job descriptions (Adzuna doesn't return structured skills)
SKILL_PATTERNS = [
    "python", "sql", "r", "java", "scala", "spark", "hadoop",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "pandas", "numpy", "matplotlib", "tableau", "power bi", "looker",
    "excel", "dbt", "airflow", "kafka", "flink", "databricks",
    "snowflake", "bigquery", "redshift", "aws", "gcp", "azure",
    "docker", "kubernetes", "git", "linux", "bash",
    "machine learning", "deep learning", "nlp", "computer vision",
    "react", "node.js", "javascript", "typescript", "html", "css",
    "mongodb", "postgresql", "mysql", "redis",
    "api", "rest", "graphql",
]

# Precompiled word-boundary regexes — one per skill.
# Plain substring matching produced false positives: "java" matched
# "javascript", "excel" matched "excellent", "rest" matched "interested".
# Lookarounds (rather than \b) handle patterns ending in non-word chars
# like "node.js" and "scikit-learn".
_SKILL_REGEXES: list[tuple[str, re.Pattern]] = [
    (s, re.compile(r"(?<![\w])" + re.escape(s) + r"(?![\w])", re.IGNORECASE))
    for s in SKILL_PATTERNS
]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

_SESSION = requests.Session()   # connection re-use across hundreds of calls

MAX_RETRIES   = 3
BACKOFF_BASE  = 2.0   # seconds: 2, 4, 8


def _get_json(url: str, *, params: dict, headers: dict | None = None,
              label: str = "API") -> dict | None:
    """
    GET a JSON endpoint with retry + exponential backoff.
    Returns the parsed dict, or None after MAX_RETRIES failures.
    4xx client errors (bad key, rate limit exhausted) are not retried.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = _SESSION.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if 400 <= status < 500 and status != 429:
                print(f"  [!] {label} HTTP {status} — not retrying: {e}")
                return None
            err = f"HTTP {status}"
        except requests.exceptions.RequestException as e:
            err = f"request error: {e}"
        except ValueError:
            err = "JSON decode error"

        if attempt < MAX_RETRIES:
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  [!] {label} {err} — retry {attempt}/{MAX_RETRIES - 1} in {wait:.0f}s")
            time.sleep(wait)
        else:
            print(f"  [!] {label} {err} — giving up after {MAX_RETRIES} attempts")
    return None


# ── Salary helpers ────────────────────────────────────────────────────────────

def _to_lpa(amount: float, currency: str = "INR", period: str = "YEAR") -> str:
    """Convert a salary figure to '₹X.X LPA' string. Returns 'Not Mentioned' if implausible."""
    if not amount or amount <= 0:
        return "Not Mentioned"

    # Convert non-INR to INR (rough exchange rates)
    fx = {"USD": 83.5, "EUR": 90.0, "GBP": 106.0, "INR": 1.0}
    inr = amount * fx.get((currency or "INR").upper(), 1.0)

    period = (period or "YEAR").upper()
    if period in ("MONTH", "MONTHLY"):
        annual_inr = inr * 12
    elif period in ("HOUR", "HOURLY"):
        annual_inr = inr * 8 * 260
    else:
        annual_inr = inr

    lpa = annual_inr / 100_000
    if lpa < 0.5 or lpa > 200:
        return "Not Mentioned"
    return f"{lpa:.1f} LPA"


def _extract_skills_from_text(text: str) -> str:
    """Scan free text for known tech skill keywords (whole-word matches only)."""
    if not text:
        return "Not Mentioned"
    found = []
    for s, pattern in _SKILL_REGEXES:
        if pattern.search(text):
            found.append(s.upper() if len(s) <= 3 else s.title())
    return ", ".join(dict.fromkeys(found)) or "Not Mentioned"   # dict preserves order, dedupes


# ── Adzuna source ─────────────────────────────────────────────────────────────

def _adzuna_fetch(keyword: str, city: str, page: int = 1, results_per_page: int = 50) -> list[dict] | None:
    """
    Fetch one page of Adzuna results for a keyword + city.
    Salary returned as min/max annual INR → normalised to LPA.
    Skills extracted from description text.

    Returns a list of jobs ([] = genuinely no more results),
    or None on request/parse error so callers can distinguish the two.
    """
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        raise EnvironmentError(
            "ADZUNA_APP_ID and ADZUNA_APP_KEY must be set in .env"
        )

    url = f"{ADZUNA_BASE}/{page}"
    params = {
        "app_id":          ADZUNA_APP_ID,
        "app_key":         ADZUNA_APP_KEY,
        "results_per_page": results_per_page,
        "what":            keyword,
        "where":           city,
        "content-type":    "application/json",
    }

    data = _get_json(url, params=params, label="Adzuna")
    if data is None:
        return None

    jobs = []
    for item in data.get("results", []):
        sal_min = item.get("salary_min") or 0
        sal_max = item.get("salary_max") or 0
        avg_sal = (sal_min + sal_max) / 2 if (sal_min or sal_max) else 0
        # Adzuna flags model-estimated salaries; optionally exclude them
        if item.get("salary_is_predicted") in (1, "1", True) and not INCLUDE_PREDICTED_SALARIES:
            avg_sal = 0
        salary  = _to_lpa(avg_sal, currency="INR", period="YEAR")

        description = item.get("description", "")
        skills      = _extract_skills_from_text(description)

        created_raw = item.get("created", "")
        try:
            date_posted = datetime.fromisoformat(
                created_raw.replace("Z", "+00:00")
            ).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_posted = "Not Mentioned"

        # Use the searched city directly — Adzuna's area[-1] gives sub-areas
        # like "HSR Layout" or "Whitefield" rather than the city itself.
        location = city

        company_obj = item.get("company", {})
        company = company_obj.get("display_name", "Not Mentioned") or "Not Mentioned"

        jobs.append({
            "job_title":   item.get("title", "Not Mentioned"),
            "company":     company,
            "location":    location,
            "experience":  "Not Mentioned",   # not in Adzuna response; imputed by pipeline
            "salary":      salary,
            "skills":      skills,
            "date_posted": date_posted,
        })

    return jobs


def _scrape_loop(
    fetch_fn,
    label: str,
    keywords: list[str],
    pages_per_keyword: int,
    delay: float,
) -> list[dict]:
    """
    Shared keyword × city × page loop for any fetch function.

    fetch_fn(keyword, city, page) must return:
      list  → jobs for that page ([] = no more results → stop paginating)
      None  → request failed after retries → skip remaining pages for this
              keyword/city but keep going with the rest of the run
    """
    all_jobs: list[dict] = []
    seen: set = set()

    print(f"[source] {label}  ({len(keywords)} keywords × {len(INDIA_CITIES)} cities × {pages_per_keyword} pages)")
    for keyword in keywords:
        for city in INDIA_CITIES:
            for page in range(1, pages_per_keyword + 1):
                print(f"  [{keyword}] {city} p{page} ...", end=" ", flush=True)
                jobs = fetch_fn(keyword, city, page=page)
                if jobs is None:          # error — already logged by _get_json
                    print("error, skipping")
                    break
                if not jobs:              # genuinely empty page
                    print("0")
                    break
                added = 0
                for job in jobs:
                    key = (job["job_title"].lower(), job["company"].lower(),
                           job.get("location", "").lower())
                    if key not in seen:
                        seen.add(key)
                        all_jobs.append(job)
                        added += 1
                print(f"+{added} new  (total: {len(all_jobs)})")
                time.sleep(delay)

    return all_jobs


def scrape_adzuna(keywords: list[str], pages_per_keyword: int, delay: float) -> list[dict]:
    """Iterate keywords × cities × pages and collect all Adzuna results."""
    return _scrape_loop(_adzuna_fetch, "Adzuna", keywords, pages_per_keyword, delay)


# ── JSearch source ────────────────────────────────────────────────────────────

def _jsearch_fetch(keyword: str, city: str, page: int = 1) -> list[dict] | None:
    """
    Fetch one page of JSearch results for a keyword + city.
    Aggregates Indeed, LinkedIn, Glassdoor job cards.

    Returns a list of jobs ([] = no more results), or None on error.
    """
    if not JSEARCH_API_KEY:
        raise EnvironmentError("JSEARCH_API_KEY must be set in .env")

    query = f"{keyword} in {city}, India"
    headers = {
        "X-RapidAPI-Key":  JSEARCH_API_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }
    params = {
        "query":     query,
        "page":      str(page),
        "num_pages": "1",
        "country":   "in",
    }

    data = _get_json(JSEARCH_BASE, params=params, headers=headers, label="JSearch")
    if data is None:
        return None

    jobs = []
    for item in data.get("data", []):
        # Salary — JSearch gives min/max + currency + period
        sal_min    = item.get("job_min_salary") or 0
        sal_max    = item.get("job_max_salary") or 0
        currency   = item.get("job_salary_currency") or "INR"
        period     = item.get("job_salary_period") or "YEAR"
        avg_sal    = (sal_min + sal_max) / 2 if (sal_min or sal_max) else 0
        salary     = _to_lpa(avg_sal, currency=currency, period=period)

        # Skills — structured list when available, else extract from description
        skills_list = item.get("job_required_skills") or []
        if skills_list:
            skills = ", ".join(skills_list)
        else:
            desc   = item.get("job_description", "")
            skills = _extract_skills_from_text(desc)

        # Experience — JSearch provides required_experience_in_months
        exp_obj     = item.get("job_required_experience") or {}
        exp_months  = exp_obj.get("required_experience_in_months")
        if exp_months:
            exp_years = round(exp_months / 12, 1)
            experience = f"{exp_years} years"
        else:
            experience = "Not Mentioned"

        # Date posted
        posted_raw = item.get("job_posted_at_datetime_utc", "")
        try:
            date_posted = datetime.fromisoformat(
                posted_raw.replace("Z", "+00:00")
            ).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_posted = "Not Mentioned"

        location = (
            item.get("job_city")
            or item.get("job_state")
            or city
        )

        jobs.append({
            "job_title":   item.get("job_title", "Not Mentioned"),
            "company":     item.get("employer_name", "Not Mentioned"),
            "location":    location,
            "experience":  experience,
            "salary":      salary,
            "skills":      skills,
            "date_posted": date_posted,
        })

    return jobs


def scrape_jsearch(keywords: list[str], pages_per_keyword: int, delay: float) -> list[dict]:
    """Iterate keywords × cities × pages and collect all JSearch results."""
    return _scrape_loop(_jsearch_fetch, "JSearch", keywords, pages_per_keyword, delay)


# ── Main entry point ──────────────────────────────────────────────────────────

def scrape(
    keywords: list[str] = None,
    pages_per_keyword: int = 2,
    output_path: str = OUTPUT_PATH,
    delay: float = 1.0,
    source: str = None,
) -> str:
    """
    Fetch India tech jobs via API and append new results to the CSV.

    Args:
        keywords:           Job search terms (defaults to DEFAULT_KEYWORDS)
        pages_per_keyword:  API pages per keyword/city pair
        output_path:        CSV file to write
        delay:              Seconds between API calls
        source:             "adzuna" or "jsearch" (overrides .env API_SOURCE)
    """
    if keywords is None:
        keywords = DEFAULT_KEYWORDS

    api = (source or API_SOURCE).lower()

    if api not in ("adzuna", "jsearch"):
        raise ValueError(f"Unknown API source '{api}'. Use 'adzuna' or 'jsearch'.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing CSV to accumulate rather than replace
    existing_jobs: list[dict] = []
    seen_keys: set = set()

    if os.path.exists(output_path):
        with open(output_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["job_title"].lower(), row["company"].lower(), row.get("location", "").lower())
                if key not in seen_keys:
                    seen_keys.add(key)
                    existing_jobs.append(row)
        print(f"[→] Loaded {len(existing_jobs):,} existing jobs from '{output_path}'")

    print(f"[→] API source  : {api.upper()}")
    print(f"[→] Keywords    : {len(keywords)}")
    print(f"[→] Cities      : {', '.join(INDIA_CITIES)}")
    print(f"[→] Pages/query : {pages_per_keyword}\n")

    # Fetch fresh jobs
    if api == "adzuna":
        new_jobs = scrape_adzuna(keywords, pages_per_keyword, delay)
    else:
        new_jobs = scrape_jsearch(keywords, pages_per_keyword, delay)

    # Merge — skip duplicates already in the CSV
    added = 0
    for job in new_jobs:
        key = (job["job_title"].lower(), job["company"].lower(), job.get("location", "").lower())
        if key not in seen_keys:
            seen_keys.add(key)
            existing_jobs.append(job)
            added += 1

    # Write
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_jobs)

    print(f"\n[✓] +{added:,} new jobs added  |  {len(existing_jobs):,} total → '{output_path}'")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch India tech jobs via Adzuna or JSearch API"
    )
    parser.add_argument(
        "--keywords", type=str, default=None,
        help="Comma-separated search terms, e.g. 'data analyst,python developer'"
    )
    parser.add_argument(
        "--pages", type=int, default=2,
        help="API pages per keyword/city pair (default: 2)"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_PATH,
        help=f"Output CSV path (default: {OUTPUT_PATH})"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        choices=["adzuna", "jsearch"],
        help="API to use — overrides API_SOURCE in .env"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between API calls (default: 1.0)"
    )
    args = parser.parse_args()

    kw_list = [k.strip() for k in args.keywords.split(",") if k.strip()] if args.keywords else None
    scrape(
        keywords=kw_list,
        pages_per_keyword=args.pages,
        output_path=args.output,
        delay=args.delay,
        source=args.source,
    )
