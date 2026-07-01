# .loop — the harness for this project

Field notes applied from LOOPS.md. The point is not ceremony; it is that the thing
which writes code never grades its own work, and state lives on disk so any session
can restart cold.

## The roles

- **Planner / you:** decide what to try next by reading `progress.md` and picking a
  `queued` item from `feature_list.json`.
- **Generator:** edits `model.py` / `data_pipeline.py` / `scraper.py` to implement it.
- **Evaluator:** `evaluate.py`. Grades the run against `contract.md`. Never trains.
  Exit 0 = SHIP, exit 1 = REJECT.

## State on disk (nothing important lives in chat)

| File | Role |
|---|---|
| `contract.md` | accept/reject gate + machine-readable thresholds |
| `feature_list.json` | candidate queue, each with a testable acceptance criterion |
| `progress.md` | current-state snapshot, rewritten each iteration |
| `../RUN_HISTORY.md` | append-only metrics log (already existed; this is the loop's log) |
| `../MODEL_ISSUES_AND_FIXES.md` | bottleneck backlog + session trace (already existed) |

## One iteration

1. Read `progress.md`; pick the next `queued` candidate.
2. Set it `in_progress` in `feature_list.json`.
3. Generate the change.
4. `python predict.py train`
5. `python .loop/evaluate.py`  ->  must print SHIP.
6. If REJECT: read the failing check, fix, retrain. If the criterion itself is
   wrong, edit `contract.md` (and say why). Do not edit metrics to pass.
7. On SHIP: mark the candidate `shipped` (or `rejected` with the result), update
   `progress.md`, append to `RUN_HISTORY.md`.

## Restart, do not patch (Rule V)

If a session goes sideways, do not keep patching it. Start fresh, read these three
files, continue. If you cannot resume from the three files, the state is too
complicated — fix that, not the session.
