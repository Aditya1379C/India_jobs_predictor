"""
log_run.py
----------
Appends a row to RUN_HISTORY.md summarizing a scrape → retrain → report
cycle (total jobs, training samples, model, test MAE, R²).

Called automatically at the end of scheduler.py and the daily-retrain
GitHub Actions workflow (train.yml) — no need to run manually.
"""

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_HERE         = Path(__file__).parent
_CSV_PATH     = _HERE / "data" / "scraped_jobs.csv"
_DB_PATH      = _HERE / "data" / "jobs.db"
_HISTORY_PATH = _HERE / "RUN_HISTORY.md"

_HEADER = (
    "# Run History\n\n"
    "Auto-generated log of each scrape → retrain → report cycle. Newest first.\n\n"
    "| Date (UTC) | Total Scraped | Jobs in DB | Training Samples | Model | Test MAE (₹ LPA) | R² | Features |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def _count_csv() -> int:
    """Count raw rows in the accumulated CSV (before any cleaning)."""
    if not _CSV_PATH.exists():
        return 0
    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus header row


def _count_db() -> int:
    """Count rows in the cleaned SQLite DB (after salary/experience parsing)."""
    if _DB_PATH.exists():
        with sqlite3.connect(_DB_PATH) as conn:
            return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    return 0


def log_run(metrics: dict) -> Path:
    """Prepend a row built from a model.py metrics dict to RUN_HISTORY.md."""
    total_scraped = _count_csv()
    jobs_in_db    = _count_db()
    date          = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    row = (
        f"| {date} | {total_scraped:,} | {jobs_in_db:,} | {metrics['n_samples']:,} | "
        f"{metrics['best_model']} | {metrics['test_mae_lpa']} | "
        f"{metrics['test_r2']} | {metrics['n_features']} |\n"
    )

    if _HISTORY_PATH.exists():
        existing = _HISTORY_PATH.read_text()
        # Find where the data rows start (after the header + divider lines)
        divider_end = existing.index("\n", existing.index("|---")) + 1
        old_data_rows = existing[divider_end:]
        # Always rewrite the header so column changes stay in sync with old data rows
        new_content = _HEADER + row + old_data_rows
    else:
        new_content = _HEADER + row

    _HISTORY_PATH.write_text(new_content)
    return _HISTORY_PATH


if __name__ == "__main__":
    import json
    metrics = json.loads((_HERE / "models" / "model_metrics.json").read_text())
    path = log_run(metrics)
    print(f"[✓] Logged run → {path}")
