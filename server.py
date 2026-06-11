"""
server.py
---------
Local HTTP server — serves the dashboard and exposes pipeline trigger endpoints.

Requirements:
    pip install flask

Usage:
    python server.py            # starts at http://localhost:8080
    python server.py --port 9090

Then open http://localhost:8080 in your browser.
Click "Run Pipeline" to trigger scrape → clean → train → report.
"""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

try:
    from flask import Flask, jsonify, send_file, abort
except ImportError:
    print("[✗] Flask is not installed. Run: pip install flask")
    sys.exit(1)

app = Flask(__name__)

# Disable noisy Flask request logging; our own logger takes over.
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Pipeline state ────────────────────────────────────────────────────────────
_state = {
    "running":      False,
    "step":         "",
    "last_message": "",
    "started_at":   None,
}
_lock = threading.Lock()

DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "report", "dashboard.html")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard HTML."""
    if not os.path.exists(DASHBOARD_PATH):
        return (
            "<p>Dashboard not generated yet. "
            "Run <code>python predict.py report</code> first.</p>",
            404,
        )
    response = send_file(DASHBOARD_PATH)
    # Prevent the browser from caching the dashboard — after a pipeline run the
    # file changes and we always want the fresh version on the next page load.
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/api/status")
def status():
    """Return the current pipeline state."""
    with _lock:
        return jsonify(dict(_state))


@app.route("/api/run", methods=["POST"])
def run():
    """Trigger the full pipeline (scrape → clean → train → report) in a background thread."""
    with _lock:
        if _state["running"]:
            return jsonify({"ok": False, "message": "Pipeline already running"}), 409
        _state["running"]  = True
        _state["step"]     = "Starting..."
        _state["started_at"] = datetime.now().isoformat()
        _state["last_message"] = ""

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": "Pipeline started"})


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _set_step(msg: str) -> None:
    with _lock:
        _state["step"] = msg
    logger.info(msg)


def _run_pipeline() -> None:
    """
    Executes scheduler.py --now as a subprocess and streams its output
    back into the step status so the dashboard can poll it.
    """
    start = time.time()
    try:
        _set_step("[1/4] Scraping new jobs…")

        # Run scheduler.py --now in the same working directory
        cwd = os.path.dirname(os.path.abspath(__file__))
        # -u: unbuffered child stdout — without it Python buffers print() output
        # when not attached to a tty, so status updates lagged until flush.
        proc = subprocess.Popen(
            [sys.executable, "-u", "scheduler.py", "--now"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Stream log lines as status updates
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            logger.info(line)
            # Map log lines to step labels
            if "[2/4]" in line:
                _set_step("[2/4] Cleaning + storing in SQLite…")
            elif "[3/4]" in line:
                _set_step("[3/4] Retraining model…")
            elif "[4/4]" in line:
                _set_step("[4/4] Regenerating dashboard…")
            else:
                with _lock:
                    _state["step"] = line[:80]   # show latest log line

        proc.wait()
        elapsed = round(time.time() - start)

        if proc.returncode == 0:
            msg = f"✓ Pipeline complete in {elapsed}s. Reload to see fresh data."
        else:
            msg = f"✗ Pipeline exited with code {proc.returncode} after {elapsed}s."

    except Exception as e:
        msg = f"✗ Pipeline error: {e}"
        logger.error(msg)

    with _lock:
        _state["running"]      = False
        _state["step"]         = ""
        _state["last_message"] = msg


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="India Jobs local server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    logger.info(f"Starting server at http://{args.host}:{args.port}")
    logger.info("Open that URL in your browser to view the dashboard.")
    logger.info("Click 'Run Pipeline' to trigger scrape → train → report.")
    logger.info("Press Ctrl+C to stop.")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
