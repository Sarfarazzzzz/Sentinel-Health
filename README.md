# Sentinel-Health — FDA Device Adverse-Event Early-Warning System

A pipeline that ingests FDA MAUDE medical device adverse-event reports, resolves
messy manufacturer records, extracts failure modes from free-text narratives, and
detects emerging safety signals using pharmacovigilance statistics (PRR, ROR, and
empirical-Bayes shrinkage).

I spent a year building clinical risk-scoring infrastructure for a medical device
manufacturer, which made me curious what you could see about device safety from
the *outside* — using only what the FDA publishes. This is that.

<!-- Add your live dashboard link here once deployed, plus a screenshot. -->
**Live dashboard:** _(deploying)_

## Results on real data

Ingested **4,594,965** MAUDE reports covering 2023-01 through 2026-06 via the
openFDA API.

| Stage | Result |
|---|---|
| Reports ingested | 4,594,965 (all unique report keys — no ingestion duplicates) |
| Manufacturer entity resolution | 7,037 raw spellings → 4,433 canonical |
| Measured duplicate rate | **5.1%** → 4,360,834 clean events |
| Narrative classification coverage | unclassified reduced **68.4% → 20.4%** across 3 taxonomy rounds |
| Extraction accuracy (300 hand-labeled reports) | keywords **47.3%**, local Qwen2.5-3B **45.0%** |
| Harm-level accuracy | keywords **65.7%**, LLM **61.7%** |

### Per-category accuracy, failure mode (n ≥ 5)

| Category | n | Keywords | LLM (3B) |
|---|---|---|---|
| implant_integration | 29 | **100%** | 72.4% |
| explant_revision | 9 | **100%** | 66.7% |
| connectivity | 11 | **100%** | 27.3% |
| battery_power | 16 | **81.2%** | 75.0% |
| mechanical_break | 24 | **79.2%** | 62.5% |
| sensor_accuracy | 25 | 68.0% | **72.0%** |
| leak_seal | 12 | 58.3% | **83.3%** |
| software | 49 | 40.8% | 40.8% |
| dosing_delivery | 21 | **23.8%** | 14.3% |
| imaging_quality | 5 | 20.0% | **80.0%** |
| cardiac_lead | 10 | 10.0% | **60.0%** |
| device_malfunction | 56 | 8.9% | 8.9% |
| alarm | 14 | 0.0% | **50.0%** |
| recall_field_action | 11 | 0.0% | 0.0% |

## Which extractor would I ship?

Neither one alone. The headline numbers are nearly tied — 47.3% for keywords
against 45.0% for the local model — but the aggregate hides the interesting part.
The two approaches fail in completely different places.

Keywords are perfect on categories where MAUDE uses boilerplate. Dental implant
reports say "FAILURE TO OSSEOINTEGRATE" and little else; a substring match gets
100% and the LLM, trying to interpret, gets 72%. Same story for explant reports
and connectivity. When the vocabulary is fixed, matching the vocabulary wins.

The LLM wins wherever the narrative has to be understood rather than scanned.
Cardiac lead reports (60% vs 10%) describe impedance readings and sensing values
in prose that shares no fixed phrasing. Alarm events are 50% vs 0%, largely
because MAUDE writes "no alert/notification occurred" and my keyword list only
looks for "alarm" — a gap I deliberately left unfixed, for reasons in the
limitations section.

So the shape I'd ship is a router, not a winner: run keywords first, accept the
result when it lands in one of the high-precision template categories, and send
everything keywords mark `unknown` to the LLM. That's implementable at runtime
because the routing decision only depends on the keyword output, not on knowing
the true label in advance. It also respects the throughput gap, which is larger
than the accuracy gap and gets ignored too often: keywords classified all 4.36M
narratives in a few minutes, while the 3B model takes a couple of seconds each —
roughly two months of compute for the same corpus. Routing only the ~20%
unclassified remainder to the model keeps the full run to hours instead of weeks.

The honest caveat is that neither approach is good enough yet on the categories
that matter most by volume. `software` sits at 40.8% for both. `device_malfunction`
is at 8.9% for both, and I think that one is partly my fault rather than the
classifiers' — see below.

## Engineering notes

Everything here was found by running the pipeline against real data, not by
planning for it.

**openFDA's pagination cap.** Every month came back with exactly 25,100 reports.
Identical numbers across different months is not a coincidence, it's a ceiling —
the API caps `skip` at 25,000, and busy months blow straight through it. The fix
was to query each window's total first and recursively split the date range in
half — month, half-month, down to single days if needed — until every piece fits
under the cap. Some February days needed splitting three levels deep.

**Disk exhaustion.** Caching raw JSON burned gigabytes per month and eventually
filled the disk mid-run, which took the laptop down with it. MAUDE records carry
a lot of nested metadata I never use. Flattening at ingest time and caching
columnar parquet per month cut storage roughly 15x.

**Memory pressure.** Concatenating every month into one DataFrame needed more RAM
than the machine has. Switched to streaming months into the silver table through
a `ParquetWriter` and truncating narratives at the silver layer, which holds
peak memory at roughly one month's worth regardless of how many years I ingest.

**Null contamination.** Eleven months into a backfill, flattening crashed on a
report with `[None]` inside `product_problems`. `dict.get(key, "")` only protects
against a missing key — an explicit null still returns `None`. Government data is
full of explicit nulls, so every join point now coerces defensively.

**Quadratic dedup.** The fuzzy duplicate pass ran fine on the synthetic test data
and then appeared to hang for hours on 4.6M rows. It was a pure-Python pairwise
loop doing pandas label lookups on every comparison. Batching each block through
`rapidfuzz.process.cdist` (multithreaded C++) with a positional numpy mask brought
the whole stage under 20 minutes.

**A dedup rate that was too good to be true.** The first clean run reported 49.8%
duplicates. Published MAUDE research puts duplication well below that, so I went
looking for my own bug instead of celebrating. First hypothesis: empty narratives
all hash identically and collapse together. Tested it — only 0.9% of narratives
are empty, so that wasn't it. The actual cause was manufacturer template text.
251,854 separate dental implant reports all read "FAILURE TO OSSEOINTEGRATE."
Those are distinct events sharing identical wording, and my dedup key was treating
every one of them as a copy of the first. I added a frequency heuristic — text
appearing more than five times is boilerplate, not a duplicate — plus event date
in the key. The corrected rate is **5.1%**, which lines up with the literature.

**A silent 100% failure rate.** My first LLM evaluation reported 1.0% accuracy on
failure mode while somehow scoring 82% on harm. That combination doesn't describe
a bad model, so I checked the output distribution: every single row was the
fallback value. A wrong model name was 404ing all 300 requests, and the `except`
block was swallowing it silently — the 82% was just the fallback happening to
match the majority harm class. I added a pre-flight check that verifies the model
exists on the server, loud error reporting, and a circuit breaker that aborts if
the first ten outputs look like fallbacks.

**Label drift in my own ground truth.** Four categories scored *identically* under
two unrelated classifiers. Two different systems failing the same way on the same
rows points at the labels, not the models. Re-reading those rows, I found
`device_malfunction` had drifted into a catch-all during labeling — I'd tagged a
report reading "the dental implant failed to osseointegrate" as
`device_malfunction` when both classifiers correctly said `implant_integration`.
My own rule was "specific mechanism wins" and I'd stopped following it somewhere
around hour three. Adjudicating those rows raised measured keyword accuracy from
38.7% to 47.3%. The classifier never changed. Only the measurement got honest.

## Known limitations

- **Keyword rules were frozen before evaluation.** I did not tune them against the
  labeled set, because tuning on your eval data inflates the number you report.
  That means known gaps stay unfixed — MAUDE writes "alert" where my taxonomy
  only matches "alarm," which is most of why `alarm` scores 0% for keywords. It's
  listed as future work rather than quietly patched.
- **The `device_malfunction` boundary is genuinely ambiguous.** When a narrative
  names no mechanism at all, whether that's the catch-all category or `unknown`
  is a definitional choice, not a fact. With a single annotator there's no
  inter-annotator agreement estimate, which is the main thing I'd add next.
- **The comparison is against a locally-served 3B model**, chosen because it's
  free and runs offline. This does not license any claim about LLMs generally; a
  frontier model would likely change the picture.
- **The empirical-Bayes layer is a Gamma-Poisson model with a single
  method-of-moments prior** — a deliberate simplification of DuMouchel's MGPS
  mixture. The shrinkage still does its main job of suppressing small-count false
  positives.
- **MAUDE is passive surveillance.** Report volume reflects reporting behavior and
  manufacturer size at least as much as device risk. Disproportionality statistics
  mitigate this by comparing within-device rates rather than raw counts, but they
  don't eliminate it. Nothing here should be read as evidence that a specific
  device is unsafe.
- **Retrospective recall validation is not yet run.** The code path exists and
  recomputes signals `as_of` each month so no future data leaks in, but the
  results aren't in this README yet.

## Architecture

```
src/ingest.py       openFDA ingestion, recursive window splitting, parquet month caches
src/dedup.py        manufacturer normalization/clustering + template-aware dedup
src/extract.py      failure-mode & harm extraction (rules | llm) + labeling loop
src/signals.py      PRR / ROR / chi2 / empirical-Bayes shrinkage + monthly trends
src/validate.py     retrospective check vs FDA recalls (no lookahead: as_of recomputation)
src/dashboard.py    Streamlit UI
run_pipeline.py     stage-by-stage orchestrator
eval_llm_ollama.py  local-LLM vs keyword evaluation against hand labels
tests/test_smoke.py offline end-to-end check with a planted signal
```

Data flows `data/raw/flat` → `data/silver` → `data/gold`. Ingestion caches one
parquet per month, so interrupted runs resume without re-downloading.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
python -m tests.test_smoke                          # offline sanity check

export OPENFDA_API_KEY=your_free_key
python run_pipeline.py --start 2025-01-01 --end 2025-03-31 --stages ingest,dedup,extract,signals
streamlit run src/dashboard.py
```

Start with one quarter before widening the window. A multi-year backfill is
several hours and a few GB of cache.

## Statistics reference

PRR and ROR are computed on the standard 2×2 contingency table per
(product_code × failure_mode). Flagging uses the Evans criteria (PRR ≥ 2,
chi² ≥ 4, N ≥ 3) combined with EB05 ≥ 1.5, where EB05 is the 5th percentile of
the Gamma-Poisson posterior — the conservative bound real signal-detection
systems alert on.

`signals.run(as_of=...)` recomputes signals using only data available at that
point in time, which is what makes the retrospective recall validation honest
rather than circular.

## Data sources

- [openFDA Device Adverse Events](https://open.fda.gov/apis/device/event/) — MAUDE reports
- [openFDA Device Recalls](https://open.fda.gov/apis/device/recall/) — validation ground truth
