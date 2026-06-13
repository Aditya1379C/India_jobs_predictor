"""
scheduler.py
------------
Auto-runner: scrape → clean → retrain → regenerate dashboard.
Default schedule: every day at 11:00 — override with --day / --time.

Setup:
    pip install schedule

Usage:
    python scheduler.py           # starts the scheduler (daily at 11:00)
    python scheduler.py --now     # run once immediately, then exit
    python scheduler.py --day friday --time 08:30   # weekly schedule

Background (Mac/Linux):
    nohup python scheduler.py > logs/scheduler.log 2>&1 &
    # To stop it: kill $(pgrep -f scheduler.py)
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import schedule

# ── Logging ───────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_LOGS_DIR = _HERE / "logs"
_LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(_LOGS_DIR / "scheduler.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    Full weekly pipeline:
      1. Scrape new jobs from the configured API source (appends to existing data)
      2. Clean + store in SQLite
      3. Retrain the salary model
      4. Regenerate the HTML dashboard
    """
    start = datetime.now()
    log.info("=" * 60)
    log.info("Weekly pipeline starting")
    log.info("=" * 60)

    # ── Step 1: Scrape ────────────────────────────────────────────────────────
    log.info("[1/4] Scraping new jobs via API...")
    try:
        from scraper import scrape, OUTPUT_PATH
        output_path = scrape(
            keywords=None,              # daily rotation batch: 10 kw × 10 cities × 2 pages
            pages_per_keyword=2,        # + category sweep = ~220 calls (within 250/day free tier)
            output_path=OUTPUT_PATH,
            delay=2.0,                  # be polite — slightly longer delay
        )
        log.info(f"  Scrape complete → {output_path}")
    except Exception as e:
        log.error(f"  Scrape failed: {e}")
        log.error("  Aborting pipeline — will retry next week.")
        return

    # ── Step 2: Clean + store ─────────────────────────────────────────────────
    log.info("[2/4] Running data pipeline (clean + store in SQLite)...")
    try:
        from data_pipeline import run
        df = run(csv_path=output_path)
        log.info(f"  Pipeline complete — {len(df):,} clean rows in DB")
    except Exception as e:
        log.error(f"  Pipeline failed: {e}")
        log.error("  Aborting pipeline — will retry next week.")
        return

    # ── Step 3: Retrain model ─────────────────────────────────────────────────
    log.info("[3/4] Retraining salary prediction model...")
    try:
        from model import train, save_model
        model, encoders, top_skills, feature_cols, metrics = train()
        save_model(model, encoders, top_skills, feature_cols, metrics)
        log.info(f"  Model retrained — winner: {metrics['best_model']}")
        log.info(f"  Test MAE: ₹{metrics['test_mae_lpa']} LPA | R²: {metrics['test_r2']}")
    except Exception as e:
        log.error(f"  Model training failed: {e}")
        # Non-fatal — keep old model, still regenerate dashboard

    # ── Step 4: Regenerate dashboard ──────────────────────────────────────────
    log.info("[4/4] Regenerating HTML dashboard...")
    try:
        from report import generate
        path = generate()
        log.info(f"  Dashboard saved → {path}")
    except Exception as e:
        log.error(f"  Dashboard generation failed: {e}")

    elapsed = round((datetime.now() - start).total_seconds())
    log.info(f"Pipeline finished in {elapsed}s")
    log.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Weekly auto-runner for the India Jobs pipeline"
    )
    parser.add_argument(
        "--now", action="store_true",
        help="Run the pipeline once immediately, then exit"
    )
    parser.add_argument(
        "--day", type=str, default="day",
        choices=["day", "monday", "tuesday", "wednesday", "thursday",
                 "friday", "saturday", "sunday"],
        help="Day of week to run, or 'day' for every day (default: day)"
    )
    parser.add_argument(
        "--time", type=str, default="11:00",
        help="Time to run in HH:MM format (default: 11:00)"
    )
    args = parser.parse_args()

    if args.now:
        log.info("--now flag detected. Running pipeline immediately.")
        run_pipeline()
        return

    # Schedule weekly run
    day_fn = getattr(schedule.every(), args.day)
    day_fn.at(args.time).do(run_pipeline)

    schedule_desc = f"every day at {args.time}" if args.day == "day" else f"every {args.day.capitalize()} at {args.time}"
    log.info(f"Scheduler started — pipeline will run {schedule_desc}")
    log.info("Press Ctrl+C to stop.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)   # check every minute
    except KeyboardInterrupt:
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
