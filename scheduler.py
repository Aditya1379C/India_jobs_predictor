"""
scheduler.py
------------
Pipeline runner: git pull → clean → retrain → regenerate dashboard.
Pulls the latest data/scraped_jobs.csv committed to GitHub by the
daily-scrape GitHub Actions workflow, then retrains.

Run manually whenever you want fresh predictions/dashboard:

Usage:
    python scheduler.py
"""

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
    Full pipeline:
      1. Git pull to fetch the latest data/scraped_jobs.csv from GitHub
      2. Clean + store in SQLite
      3. Retrain the salary model
      4. Regenerate the HTML dashboard
    """
    start = datetime.now()
    log.info("=" * 60)
    log.info("Pipeline starting")
    log.info("=" * 60)

    # ── Step 1: Git pull ──────────────────────────────────────────────────────
    log.info("[1/4] Pulling latest data from GitHub...")
    output_path = str(_HERE / "data" / "scraped_jobs.csv")
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=_HERE, capture_output=True, text=True, check=True,
        )
        log.info(f"  {result.stdout.strip() or 'Already up to date.'}")
    except subprocess.CalledProcessError as e:
        log.error(f"  Git pull failed: {e.stderr.strip()}")
        log.error("  Aborting pipeline — will retry next run.")
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
    metrics = None
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

    # ── Step 5: Log run summary ───────────────────────────────────────────────
    if metrics is not None:
        try:
            from log_run import log_run
            history_path = log_run(metrics)
            log.info(f"  Run logged → {history_path}")
        except Exception as e:
            log.error(f"  Run logging failed: {e}")

    elapsed = round((datetime.now() - start).total_seconds())
    log.info(f"Pipeline finished in {elapsed}s")
    log.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    run_pipeline()


if __name__ == "__main__":
    main()
