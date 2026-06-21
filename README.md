# India Tech Jobs — Salary Predictor

An end-to-end Python pipeline that fetches India tech job listings via API, cleans and stores them in SQLite, trains a salary prediction ML model (Random Forest + XGBoost), and generates an interactive HTML dashboard — all runnable from a single CLI.

---

## How It Works

```
scraper.py  →  data/scraped_jobs.csv
                      ↓
           data_pipeline.py  →  data/jobs.db (SQLite)
                                       ↓
                                  model.py  →  models/
                                                   ↓
                                            report.py  →  report/dashboard.html
```

`scheduler.py` runs this entire chain in one shot (git pull → clean → retrain → report). The daily scrape itself runs on GitHub Actions (`.github/workflows/scrape.yml`); run `scheduler.py` locally whenever you want to pull the latest data and retrain.

---

## Project Structure

```
india_jobs_predictor/
├── .github/workflows/      # Daily cloud scrape + retrain + CI test runs
├── data/                   # Raw CSV + SQLite database (auto-created)
├── models/                 # Trained model, encoders, metrics (auto-created)
├── report/                 # Generated HTML dashboard (auto-created)
├── logs/                   # Scheduler logs (auto-created)
├── scraper.py              # Fetch jobs via Adzuna or JSearch API
├── data_pipeline.py        # Clean, parse salary/experience, store in SQLite
├── model.py                # Train RF + XGBoost, group-mean target encoding, predict
├── predict.py              # CLI entry point
├── report.py               # Chart.js HTML dashboard generator
├── scheduler.py            # One-shot pipeline runner (git pull → clean → retrain → report)
├── log_run.py              # Appends a metrics summary row to RUN_HISTORY.md
├── server.py               # Local Flask dashboard server
├── tests/                  # Unit tests for parsing & feature logic
├── RUN_HISTORY.md          # Auto-logged metrics from each pipeline run
├── .env.example            # API key template — copy to .env
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your key(s). Two free API options:

| API | Free tier | Sign up |
|-----|-----------|---------|
| **Adzuna** | 250 calls/day | https://developer.adzuna.com |
| **JSearch** (RapidAPI) | 500 calls/month — aggregates Indeed, LinkedIn, Glassdoor | https://rapidapi.com/letscrape-6bfp2kogce/api/jsearch |

Set `API_SOURCE=adzuna` or `API_SOURCE=jsearch` in `.env`.

---

## Usage

### Scrape fresh job data

```bash
python predict.py scrape
python predict.py scrape --pages 5          # more pages per keyword
```

Fetches listings for a rotating daily batch of keywords (10/day from a 36-keyword pool, full coverage every 4 days) × 10 India cities, plus a broad `it-jobs` category sweep — all within Adzuna's 250-calls/day free tier. New results append to `data/scraped_jobs.csv`. Pass `--keywords "a,b,c"` to override the rotation.

### Load from an existing CSV

```bash
python predict.py pipeline --csv data/scraped_jobs.csv
```

### Train the model

```bash
python predict.py train
```

Output:
```
Best model : Random Forest
Test MAE   : ₹4.05 LPA
R²         : 0.315
Samples    : 3,573
```

Compares Random Forest and XGBoost via 5-fold cross-validation and saves the winner to `models/`. Metrics are honest held-out test scores — target encoders and outlier fences are fit on the training split only, so there's no leakage inflating the numbers (an earlier version scored a flattering but fake R² of 0.88 before the leakage was fixed).

### Generate the dashboard

```bash
python predict.py report
```

Opens at `report/dashboard.html`. Six sections: city breakdown, role distribution, skills, salary analysis, scatter (experience vs salary), and an interactive salary predictor.

### Predict a salary

```bash
python predict.py salary --role "Data Analyst" --city "Bangalore" --exp 3
python predict.py salary --role "ML Engineer" --city "Hyderabad" --exp 5 --skills "Python,PyTorch"
```

Output:
```
╭────────────────────────────┬──────────────────────╮
│ Metric                     │ Value                │
├────────────────────────────┼──────────────────────┤
│ Predicted Salary           │ ₹ 9.4 LPA            │
│ Range (10th–90th %ile)     │ ₹ 7.2 – 11.8 LPA     │
╰────────────────────────────┴──────────────────────╯
```

### Run the full pipeline once (manual trigger)

```bash
python scheduler.py
```

### Run the tests

```bash
make test        # or: python -m pytest tests/ -v
```

47 unit tests cover the fragile parsing logic: salary strings, experience ranges, skill extraction, title normalization, keyword rotation, and the API call budget.

---

## Automation

The full pipeline runs entirely on GitHub's servers — your machine can be off:

1. **`scrape.yml`** — scrapes daily at 11:00 IST, commits the updated `data/scraped_jobs.csv`. Requires `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` as repo secrets.
2. **`train.yml`** — triggers automatically once `scrape.yml` completes: cleans the data, retrains the model, regenerates the dashboard, logs a summary row to [`RUN_HISTORY.md`](RUN_HISTORY.md), and commits `report/dashboard.html` + `models/*.pkl` + `models/*.json` back to the repo (force-added despite `.gitignore` — see the note there).

So all you need to do is pull and view:
```bash
git pull
python server.py        # http://127.0.0.1:8080
```
Check [`RUN_HISTORY.md`](RUN_HISTORY.md) for a quick log of every scrape → retrain → report cycle (total jobs, training samples, model, test MAE, R²) without needing to open the dashboard.

`scheduler.py` still exists for running the same pipeline locally on demand (git pull → clean → retrain → report → log), e.g. if you want to retrain without waiting for the daily cron.

CI (`.github/workflows/ci.yml`) runs the test suite on every push.

---

## ML Details

| | |
|---|---|
| **Algorithms** | Random Forest, XGBoost (best selected by 5-fold CV MAE) |
| **Target** | Log-transformed salary (LPA) — reverses with `expm1` at inference |
| **Encoding** | Group-mean target encoding (fit on train split only) for `job_title`, `city`, `company` |
| **Features** | Experience years, seniority, company tier, city, company, job title, top-20 skill flags |
| **Confidence interval** | Random Forest: 10th–90th percentile across trees · XGBoost: ±15% |

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data cleaning & transformation |
| `sqlite3` | Local database (history preserved across runs) |
| `scikit-learn` | Random Forest, preprocessing, cross-validation |
| `xgboost` | XGBoost regressor |
| `requests` + `python-dotenv` | API calls, environment config |
| `typer` + `rich` | CLI interface |
| Chart.js 4 | Interactive dashboard charts (frontend, no install needed) |

---

## Skills Demonstrated

- API integration with retry/backoff, call budgeting, and keyword rotation within a free-tier quota
- Data cleaning, feature engineering, salary/experience parsing (unit-tested)
- Machine learning: model selection, cross-validation, log-target regression
- Leakage-safe target encoding for high-cardinality categoricals (split-first, fit on train only)
- Automation: GitHub Actions cloud scraping + CI, local Flask dashboard server
- Interactive HTML dashboard with dark mode, Chart.js, CSS custom properties
- CLI tool design with typer
