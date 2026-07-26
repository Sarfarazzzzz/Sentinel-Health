# Sentinel-Health — FDA Device Adverse-Event Early-Warning System

Ingests FDA MAUDE adverse-event reports monthly, deduplicates the notoriously
messy records, extracts failure modes from free-text narratives, detects
emerging safety signals with real pharmacovigilance statistics (PRR, ROR,
empirical-Bayes shrinkage), and validates itself retrospectively against
actual FDA recalls.

## Data sources (all free, no scraping)

| Source | What | URL |
|---|---|---|
| openFDA Device Events | MAUDE adverse-event reports (the core feed) | https://open.fda.gov/apis/device/event/ |
| openFDA Device Recalls | Ground truth for validation | https://open.fda.gov/apis/device/recall/ |
| API key (optional, free) | Raises rate limits substantially | https://open.fda.gov/apis/authentication/ |

No key is needed to start. Get one before ingesting multi-year windows.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENFDA_API_KEY=your_key        # optional but recommended
export ANTHROPIC_API_KEY=your_key      # only for --extract-mode llm
```

## Verify everything works before touching the network

```bash
python -m tests.test_smoke
```

This builds synthetic data with a planted device defect and asserts the full
dedup → extract → signal chain finds it (and nothing else).

## Run the real pipeline

```bash
# Start small: one quarter of data, rules-based extraction, no LLM cost.
python run_pipeline.py --start 2025-01-01 --end 2025-03-31 --stages ingest,dedup,extract,signals

# Then widen the window. 2–3 years of data makes signals meaningful.
python run_pipeline.py --start 2023-01-01 --end 2026-06-30

# Retrospective validation (slow — recomputes signals month by month, honestly)
python run_pipeline.py --stages validate

# Dashboard
streamlit run src/dashboard.py
```

Ingestion is cached per month under `data/raw/`, so re-runs are cheap and
resumable. Data layout: `data/raw` (JSON as fetched) → `data/silver`
(flattened, deduped parquet) → `data/gold` (extracted, signals, validation).

## The parts that are deliberately yours to do

This scaffold runs, but it is a starting point, not a finished portfolio piece.
The value — and your interview defensibility — comes from:

1. **Tune the dedup thresholds** (`src/dedup.py`). Pull 50 flagged duplicate
   pairs, read them, and decide whether 95 is right. Record your dedup rate.
2. **Grow the failure-mode taxonomy** (`src/extract.py`). The starter keywords
   are naive. Read a few hundred real narratives; the taxonomy you build from
   them is domain knowledge no one can copy.
3. **Hand-label 300 narratives** (`python -m src.extract --make-labels 300`),
   then evaluate rules vs. LLM extraction (`--evaluate data/labels_todo.csv`)
   and write up which you'd ship and why, with the cost/accuracy numbers.
4. **Interrogate the validation misses.** For every recall the system did not
   flag, figure out why (no reports? wrong product code mapping? taxonomy
   gap?) and write it down. This section of your writeup will do more work in
   interviews than any hit-rate number.
5. **Productionize**: wrap `run_pipeline.py` in an Airflow/Prefect monthly DAG,
   add Great Expectations checks between layers, Dockerize, track extraction
   model versions in MLflow. You have all of these on your resume — make the
   repo prove it.

## Statistics notes (know these cold before an interview)

- PRR/ROR are computed on the standard 2×2 contingency table per
  (product_code × failure_mode); flagging uses the Evans criteria
  (PRR ≥ 2, chi² ≥ 4, N ≥ 3) plus EB05 ≥ 1.5.
- The empirical-Bayes layer is a Gamma-Poisson model with a single
  method-of-moments Gamma prior — a deliberate simplification of DuMouchel's
  MGPS mixture. Be ready to explain what the shrinkage buys you (kills
  small-count false positives) and what the full mixture would add.
- `signals.run(as_of=...)` recomputes signals using only data available at
  that time; the validator uses it so no future information leaks. If an
  interviewer asks how you avoided lookahead bias, this is the answer.
