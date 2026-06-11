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

The scheduler (`scheduler.py`) runs this entire chain automatically — daily at 11:00 by default, configurable with `--day`/`--time`.

---

## Project Structure

```
india_jobs_predictor/
├── data/                   # Raw CSV + SQLite database (auto-created)
├── models/                 # Trained model, encoders, metrics (auto-created)
├── report/                 # Generated HTML dashboard (auto-created)
├── logs/                   # Scheduler logs (auto-created)
├── scraper.py              # Fetch jobs via Adzuna or JSearch API
├── data_pipeline.py        # Clean, parse salary/experience, store in SQLite
├── model.py                # Train RF + XGBoost, group-mean target encoding, predict
├── predict.py              # CLI entry point
├── report.py               # Chart.js HTML dashboard generator
├── scheduler.py            # Auto-runner (daily by default)
├── server.py               # Local Flask dashboard server
├── tests/                  # Unit tests for parsing & feature logic
├── .env.example            # API key template — copy to .env
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install pandas scikit-learn xgboost requests python-dotenv schedule typer rich
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

Fetches listings for 18 default keywords × 6 India cities and appends new results to `data/scraped_jobs.csv`.

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
Best model : RandomForest
Test MAE   : ₹2.1 LPA
R²         : 0.71
Samples    : 3,842
```

Compares Random Forest and XGBoost via 5-fold cross-validation and saves the winner to `models/`.

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
python scheduler.py --now
```

### Schedule auto-runs

```bash
python scheduler.py                          # every day at 11:00
python scheduler.py --day friday --time 08:30
nohup python scheduler.py > logs/scheduler.log 2>&1 &   # background
```

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
| `schedule` | Weekly pipeline scheduler |
| `typer` + `rich` | CLI interface |
| Chart.js 4 | Interactive dashboard charts (frontend, no install needed) |

---

## Skills Demonstrated

- API integration and data collection pipeline
- Data cleaning, feature engineering, salary/experience parsing
- Machine learning: model selection, cross-validation, log-target regression
- Leakage-safe target encoding for high-cardinality categoricals (split-first, fit on train only)
- Automated scheduler + local Flask dashboard server
- Interactive HTML dashboard with dark mode, Chart.js, CSS custom properties
- CLI tool design with typer
