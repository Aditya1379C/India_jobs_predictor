#!/usr/bin/env python3
"""Evaluator role for the loop. Grades a model run against .loop/contract.md.

This is deliberately a SEPARATE program from model.py. The thing that trains a
model must never be the thing that decides the model is good enough (Rule II:
separate the roles). Run this after `python predict.py train`:

    python .loop/evaluate.py            # grade the committed model_metrics.json
    python .loop/evaluate.py --no-tests # skip the pytest gate (faster, less strict)

Exit code 0 = SHIP, 1 = REJECT. Wire it into CI or a loop so a failing run
cannot be committed. Reads thresholds only from the JSON block in contract.md,
so there is a single source of truth for "done".
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / ".loop" / "contract.md"
METRICS = ROOT / "models" / "model_metrics.json"


def load_gate() -> dict:
    text = CONTRACT.read_text(encoding="utf-8")
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        sys.exit("contract.md has no ```json gate block; cannot grade.")
    return json.loads(match.group(1))


def load_metrics() -> dict:
    if not METRICS.exists():
        sys.exit(f"{METRICS} not found. Run `python predict.py train` first.")
    return json.loads(METRICS.read_text(encoding="utf-8"))


def run_tests() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    return proc.returncode == 0, tail[-1] if tail else "no pytest output"


def main() -> int:
    skip_tests = "--no-tests" in sys.argv
    gate = load_gate()
    m = load_metrics()

    r2 = m.get("test_r2")
    mae = m.get("test_mae_lpa")
    n = m.get("n_samples")
    by_source = m.get("metrics_by_source") or {}

    checks: list[tuple[str, bool, str]] = []

    checks.append((
        "R2 floor",
        r2 is not None and r2 >= gate["min_r2_overall"],
        f"test_r2={r2} (need >= {gate['min_r2_overall']})",
    ))
    checks.append((
        "MAE ceiling",
        mae is not None and mae <= gate["max_mae_lpa"],
        f"test_mae_lpa={mae} (need <= {gate['max_mae_lpa']})",
    ))
    checks.append((
        "sample floor",
        n is not None and n >= gate["min_samples"],
        f"n_samples={n} (need >= {gate['min_samples']})",
    ))
    checks.append((
        "no leakage (overall)",
        r2 is not None and r2 <= gate["leakage_ceiling_r2"],
        f"test_r2={r2} (leakage if > {gate['leakage_ceiling_r2']})",
    ))

    for src, stats in by_source.items():
        s_r2 = stats.get("r2") if isinstance(stats, dict) else None
        if s_r2 is not None:
            checks.append((
                f"no leakage ({src})",
                s_r2 <= gate["per_source_leakage_ceiling_r2"],
                f"{src} r2={s_r2} (leakage if > {gate['per_source_leakage_ceiling_r2']})",
            ))

    if gate.get("require_metrics_by_source"):
        checks.append((
            "per-source reported",
            bool(by_source),
            "metrics_by_source present" if by_source else "metrics_by_source MISSING",
        ))

    if gate.get("require_tests_pass") and not skip_tests:
        ok, summary = run_tests()
        checks.append(("tests pass", ok, summary))

    print("\n  Loop evaluator — grading against contract.md\n")
    all_ok = True
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:24} {detail}")
        all_ok = all_ok and ok

    verdict = "SHIP" if all_ok else "REJECT"
    print(f"\n  Verdict: {verdict}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
