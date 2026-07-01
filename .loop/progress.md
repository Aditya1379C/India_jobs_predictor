# Progress snapshot

Rewritten each iteration. The three-file test: a fresh session should be able to
resume from this file plus contract.md plus feature_list.json, without reading the
chat. If it cannot, this file is not describing enough state.

## Where we are (2026-07-01)

- **Best model to date (in-memory experiments):** XGBoost blend, overall test R2
  **0.420** / MAE **3.75 LPA** / 10,719 samples. Per-source: scrape 0.342, kaggle 0.514.
  Band heads: min R2 0.470, max R2 0.569.
- **What is actually committed on disk:** `models/model_metrics.json` shows R2
  **0.171** / MAE **4.32** / 7,443 samples / 28 features. This is a scrape-only run
  and is a REGRESSION from the blended best. The blended experiments ran with
  artifacts untouched, so the good model was never saved.
- **Immediate consequence:** running `python .loop/evaluate.py` right now REJECTS
  (0.171 < 0.30 floor). This is the gate doing its job, not a bug.

## Next action

Reproduce the blended run and commit it, so the on-disk model matches the best
known result, then re-grade:

1. Ensure the Kaggle real-band rows are in the DB (`data/jobs.db`; backup exists at
   `data/jobs.db.bak-pre-kaggle`).
2. `python predict.py train`  (writes fresh model_metrics.json)
3. `python .loop/evaluate.py`  (must print SHIP)
4. `python predict.py report`  then append the row to RUN_HISTORY.md via log_run.py
5. Only then pick up the next `queued` item in feature_list.json.

## Bottleneck right now (Rule IX)

Data, not model. Documented ceiling ~0.5 with real bands, ~0.35 with quantized
Adzuna. The next real gain is more disclosed-salary rows (see candidate
`more-real-band-sources`), not more feature engineering.
