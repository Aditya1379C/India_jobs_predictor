# India Tech Jobs - Salary Predictor

An end-to-end Python data pipeline that scrapes India tech job listings via public APIs, cleans and stores them in SQLite, trains a salary prediction model (Random Forest + XGBoost), and serves an interactive HTML dashboard - fully automated on GitHub Actions and runnable from a single CLI.

> **Current model:** XGBoost · MAE ₹4.21 LPA · R² 0.243 · 3,263 training samples

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-automated-2088FF?logo=github-actions&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What Makes This Different

Most salary data projects work off a static Kaggle CSV. This one pulls **live data** from real job APIs every day, processes it through a production-style engineering pipeline, trains a model with proper ML hygiene, and deploys a dashboard - the same workflow you'd build at a company.

---

## How It Works

```
APIs (Adzuna / JSearch)
        │
        ▼
   scraper.py  ───────────────────►  data/scraped_jobs.csv
        │
        ▼
data_pipeline.py  ──────────────────►  data/jobs.db  (SQLite)
        │
        ▼
   model.py  ───────────────────────►  models/salary_model.pkl
        │                                      + feature_importance.json
        ▼                                      + model_metrics.json
   report.py  ──────────────────────►  report/dashboard.html
        │
        ▼
   server.py  ──────────────────────►  http://127.0.0.1:8080
```

`scheduler.py` runs the full chain in one shot: git pull → clean → retrain → report → log. The daily scrape runs automatically on GitHub Actions; pull locally whenever you want the latest data.

---

## Project Structure

```
india_jobs_predictor/
├── .github/workflows/      # Daily cloud scrape, auto-retrain, CI test runs
├── data/                   # Raw CSV + SQLite database (auto-created)
├── models/                 # Trained model, encoders, feature importance (auto-created)
├── report/                 # Generated HTML dashboard (auto-created)
├── tests/                  # 42 unit tests for parsing and feature logic
├── scraper.py              # Fetch jobs via Adzuna or JSearch API
├── data_pipeline.py        # Clean, parse salary/experience, write to SQLite
├── model.py                # Train RF + XGBoost, group-mean encoding, predict
├── predict.py              # CLI entry point (scrape / pipeline / train / report / salary / serve)
├── report.py               # Chart.js HTML dashboard generator
├── scheduler.py            # One-shot local pipeline runner
├── server.py               # Local Flask dashboard server
├── log_run.py              # Appends metrics row to RUN_HISTORY.md after each run
├── RUN_HISTORY.md          # Auto-logged history of every scrape → retrain → report cycle
├── Makefile                # Common task shortcuts
└── .env.example            # API key template - copy to .env
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

Two free API options - pick one:

| API | Free tier | Sign up |
|---|---|---|
| **Adzuna** | 250 calls/day | https://developer.adzuna.com |
| **JSearch** (RapidAPI) | 500 calls/month · aggregates Indeed, LinkedIn, Glassdoor | https://rapidapi.com/letscrape-6bfp2kogce/api/jsearch |

Set `API_SOURCE=adzuna` or `API_SOURCE=jsearch` in `.env`.

---

## Usage

### Scrape fresh job data

```bash
python predict.py scrape
python predict.py scrape --pages 5          # more pages per keyword
```

Fetches listings across a rotating batch of 10 keywords/day (36-keyword pool, full cycle every 4 days) × 10 India cities - all within Adzuna's 250-call free tier. Results append to `data/scraped_jobs.csv`.

### Train the model

```bash
python predict.py train
```

Compares Random Forest and XGBoost via 5-fold cross-validation and saves the winner:

```
Best model : XGBoost
Test MAE   : ₹4.21 LPA
R²         : 0.243
Samples    : 3,263
```

Target encoders and outlier fences are fit on the training split only - no leakage. An earlier version scored a misleading R² of 0.88 before this was fixed.

### Generate the dashboard

```bash
python predict.py report
# open report/dashboard.html
```

Six sections: city breakdown, role distribution, skills heatmap, salary analysis, experience vs. salary scatter, and an interactive salary predictor.

### Predict a salary

```bash
python predict.py salary --role "Data Analyst" --city "Bangalore" --exp 3
python predict.py salary --role "ML Engineer"  --city "Hyderabad" --exp 5 --skills "Python,PyTorch"
```

```
╭────────────────────────────┬──────────────────────╮
│ Metric                     │ Value                │
├────────────────────────────┼──────────────────────┤
│ Predicted Salary           │ ₹ 9.4 LPA            │
│ Range (10th–90th %ile)     │ ₹ 7.2 – 11.8 LPA     │
╰────────────────────────────┴──────────────────────╯
```

### Serve the dashboard locally

```bash
python predict.py serve     # http://127.0.0.1:8080
```

### Run the tests

```bash
make test        # or: python -m pytest tests/ -v
```

42 unit tests cover salary string parsing, experience range extraction, skill regex matching, title normalisation, keyword rotation, and API call budgeting.

---

## Automation

The pipeline runs entirely on GitHub - your machine can be off:

| Workflow | Trigger | What it does |
|---|---|---|
| `scrape.yml` | Daily 11:00 IST | Scrapes jobs, commits updated CSV |
| `train.yml` | After `scrape.yml` completes | Cleans data, retrains model, regenerates dashboard, logs to `RUN_HISTORY.md`, commits model + dashboard back to repo |
| `ci.yml` | Every push | Runs the full test suite |

To use the latest data locally:
```bash
git pull
python predict.py serve     # http://127.0.0.1:8080
```

[`RUN_HISTORY.md`](RUN_HISTORY.md) gives a quick log of every cycle (total jobs, training samples, model, MAE, R²) without opening the dashboard.

---

## ML Details

| | |
|---|---|
| **Algorithms** | Random Forest, XGBoost - best chosen by 5-fold CV MAE |
| **Target variable** | Log-transformed salary (LPA), reversed with `expm1` at inference |
| **Encoding** | Group-mean target encoding for `job_title`, `city`, `company` - fit on training split only |
| **Features** | Experience years, seniority level, company tier, city, job title, top-20 skill flags (26 total) |
| **Confidence interval** | Random Forest: 10th–90th percentile across trees · XGBoost: ±15% fallback |
| **Leakage guard** | `train_test_split` first, then encoders and outlier fences fit exclusively on `df_train` |

---

## Libraries

| Library | Purpose |
|---|---|
| `pandas` | Data cleaning and transformation |
| `sqlite3` | Persistent local job database |
| `scikit-learn` | Random Forest, preprocessing, cross-validation |
| `xgboost` | XGBoost regressor |
| `requests` + `python-dotenv` | API calls and environment config |
| `typer` + `rich` | CLI interface and formatted output |
| `flask` | Local dashboard server |
| Chart.js 4 | Interactive frontend charts (no install needed) |

---

## Skills Demonstrated

- **API integration** - retry/backoff, call budget guard, daily keyword rotation within a free-tier quota
- **Data engineering** - multi-format salary/experience parsing, city normalisation, incremental SQLite upsert (unit-tested)
- **Machine learning** - model selection via cross-validation, log-target regression, leakage-safe target encoding for high-cardinality categoricals
- **Automation** - GitHub Actions cloud pipeline (scrape → retrain → commit), local Flask server
- **Frontend** - self-contained HTML dashboard, dark mode, Chart.js, CSS custom properties, interactive salary predictor
- **Software engineering** - CLI design with Typer, 42 unit tests, clean separation of scraping / pipeline / model / report layers
