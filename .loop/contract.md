# Contract: salary-model iteration

The accept/reject gate for any change to scraping, features, or the model. A run
is **shippable** only if every assertion below holds. This file is the thing that
gets graded. The generator (whoever edits `model.py` / `data_pipeline.py` /
`scraper.py`) does NOT decide pass/fail. `evaluate.py` reads this file and grades
the run. Change the criteria by editing this file, never by editing a metric to fit.

Negotiate here first, then generate. If a change cannot pass this contract, either
the change is wrong or the contract is wrong. Fix the contract only when the
criterion itself is wrong, not when a build fails it.

## What "done" means for one iteration

1. **No regression.** Overall held-out test R2 does not fall below the floor, and
   MAE does not rise above the ceiling (see gate below).
2. **No leakage.** Test R2 stays under the leakage ceiling. The documented data
   ceiling for this problem is ~0.5 (real bands) and ~0.35 for quantized Adzuna
   data. Any overall or per-source R2 above the ceiling is treated as probable
   leakage until proven otherwise, exactly like the LOO-encoding 0.88 incident.
3. **Split-first discipline holds.** Encoders and outlier fences are fit on the
   training split only. This is asserted in tests, not eyeballed.
4. **Tests pass.** The full `pytest` suite is green.
5. **Provenance reported.** If the DB carries a `source` column, the run reports
   per-source R2 (scrape vs kaggle vs any new source), so a good overall number
   cannot hide a broken slice.
6. **State is written to disk.** `models/model_metrics.json` reflects the run being
   graded, and the run is appended to `RUN_HISTORY.md`.

## What we are NOT changing (settled, do not relitigate)

Carried over from MODEL_ISSUES_AND_FIXES.md so the loop does not rediscover dead ends:

- Group-mean target encoding stays (LOO failed: 0.88 train / 0.09 test).
- Company encoder hard cutoff `MIN_ENCODER_SAMPLES=3` stays (smoothing lost every sweep).
- Split-first, then fit encoders/fences on train only. Must not regress.
- 5-fold CV for model selection stays.
- `REAL_BAND_WEIGHT=3.0` (x3 up-weight on non-scrape rows) is the tuned sweet spot.

## Machine-readable gate

`evaluate.py` parses the JSON block below. Tune numbers here; the evaluator has no
thresholds of its own.

```json
{
  "min_r2_overall": 0.30,
  "max_mae_lpa": 4.50,
  "min_samples": 3000,
  "leakage_ceiling_r2": 0.70,
  "per_source_leakage_ceiling_r2": 0.85,
  "require_tests_pass": true,
  "require_metrics_by_source": false
}
```

### Why these numbers

- `min_r2_overall 0.30`: healthy blended runs hit 0.40+, and even scrape-heavy runs
  reach ~0.32. Below 0.30 something regressed (the committed 0.171 scrape-only model
  fails this on purpose: it is the regression the gate is meant to catch).
- `max_mae_lpa 4.50`: best is 3.75; 4.50 leaves headroom for data drift without
  waving through a broken run.
- `leakage_ceiling_r2 0.70`: comfortably above the ~0.5 real-data ceiling, so a jump
  to 0.7+ means the target leaked into features, not that the model got smart.
- `require_metrics_by_source false`: flip to true once every run reliably emits
  `metrics_by_source`, to make provenance a hard requirement.
