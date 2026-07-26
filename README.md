# Sentinel-Health — FDA Device Adverse-Event Early-Warning System

A pipeline that ingests FDA MAUDE medical device adverse-event reports, resolves
messy manufacturer records, extracts failure modes from free-text narratives, and
detects emerging safety signals — with reporting-artifact detection as a
first-class feature rather than an afterthought.

I spent a year building clinical risk-scoring infrastructure for a medical device
manufacturer, which made me curious what you could see about device safety from
the *outside*, using only what the FDA publishes. This is that.

**Live dashboard:** <!-- ['Dashboard Link'](https://sentinel-health-qraq8yvjf52zyehxtfmm2b.streamlit.app/) -->

## What it found

Every alert investigated so far traced to **reporting behaviour rather than device
behaviour** — a manufacturer filing a backlog, changing narrative wording, or
routing service-centre returns to MAUDE for the first time.

That is the central finding, not a failure. Passive surveillance data reflects how
events are reported at least as strongly as how often they occur, so a system built
on it needs artifact detection built in. Four worked investigations, with raw
output and adjudications, are in [`docs/investigations.md`](docs/investigations.md):

| Alert | Verdict |
|---|---|
| CGM software, 2026-01 (57k reports in one month) | Batch retrospective filing of a real coding defect |
| Infusion pump sensor accuracy, 2026-04 | New filing practice at one manufacturer **+ a taxonomy precedence bug** |
| Ventilator contamination, 2026-02 | Philips foam-recall remediation returns, misclassified |
| Endoscope leak-seal across 5 device types, 2026-04/05 | Coordinated Olympus filing change — and a gap in my own artifact guard |

## Results on real data

Ingested **4,594,965** MAUDE reports covering 2023-01 → 2026-06 via the openFDA API.

| Stage | Result |
|---|---|
| Reports ingested | 4,594,965 (all unique report keys — no ingestion duplicates) |
| Manufacturer entity resolution | 7,037 raw spellings → 4,433 canonical |
| Measured duplicate rate | **5.1%** → 4,360,834 clean events |
| Narrative classification coverage | unclassified reduced **68.4% → 20.4%** over 3 taxonomy rounds |
| Extraction accuracy (300 hand-labeled reports) | keywords **47.3%**, local Qwen2.5-3B **45.0%** |
| Harm-level accuracy | keywords **65.7%**, LLM **61.7%** |
| Active alerts (trailing 6 months) | 98, of which 65 show no artifact pattern |

### Per-category extraction accuracy (failure mode, n ≥ 5)

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

## Which extractor would I ship?

Neither alone. The headline numbers are nearly tied, but the aggregate hides the
interesting part: the two approaches fail in completely different places.

Keywords are perfect where MAUDE uses boilerplate. Dental implant reports say
"FAILURE TO OSSEOINTEGRATE" and little else; substring matching gets 100% and the
LLM, trying to interpret, gets 72%. When the vocabulary is fixed, matching the
vocabulary wins. The LLM wins where narratives must be understood rather than
scanned — cardiac lead reports (60% vs 10%) describe impedance and sensing values
in prose with no fixed phrasing.

So the shape I would ship is a router: run keywords first, accept the result in
the high-precision template categories, and send everything keywords mark
`unknown` to the LLM. That is implementable at runtime because the routing
decision depends only on the keyword output, not on knowing the true label. It
also respects the throughput gap, which is larger than the accuracy gap: keywords
classified all 4.36M narratives in minutes, while the 3B model takes seconds each
— roughly two months of compute for the same corpus. Routing only the ~20%
unclassified remainder keeps a full run to hours.

The honest caveat is that neither is good enough on the categories that matter
most by volume. `software` sits at 40.8% for both, and `device_malfunction` at
8.9% for both — partly my own labelling, discussed below.

## How the signal layer got here

**v1 — cross-sectional, all devices.** Top signal: a heating pad with electrical
failures. That is what a heating pad *is*, not a discovery.

**v2 — stratified by medical specialty panel.** Better, but still tautologies
(root canal resin over-reports sealing failures), and PRR values reached 80,000:
panels are dominated by one device type — the Dental panel is ~95% implant
osseointegration reports — so every other failure mode has a near-zero baseline
and any device specialising in it divides by ~nothing.

The underlying problem: cross-sectional disproportionality assumes devices within
a comparison group are exchangeable. Medical devices are not — each type has
characteristic failure modes by design, so "which device over-reports mode Y"
largely recovers "which device is built to do Y." No threshold fixes that.

**v3 — self-controlled temporal detection.** The right question for an early
warning is not "more than its peers?" but "more than it used to?" Each device
becomes its own control, device-type confounding cancels by construction, and
what surfaces is change rather than identity. Disproportionality was kept as
descriptive context, with a minimum expected count to kill degenerate baselines.

**v4 — corrections from investigating v3's own alerts.** Recall-remediation
reports excluded from alerting (a spike means a recall campaign is *already*
running). Minimum monthly reports raised from 5 to 25 after alerts with n=6 were
producing z-scores above 20 on pure denominator noise.

## Engineering notes

Everything here was found by running the pipeline against real data.

**openFDA's pagination cap.** Every month returned exactly 25,100 reports.
Identical numbers across different months is a ceiling, not a coincidence — the
API caps `skip` at 25,000. Fixed by querying each window's total first and
recursively splitting the date range until every piece fits. Some February days
needed splitting three levels deep.

**Disk exhaustion.** Caching raw JSON burned gigabytes per month and eventually
filled the disk mid-run. Flattening at ingest time and caching columnar parquet
per month cut storage roughly 15x.

**Memory pressure.** Concatenating every month into one DataFrame needed more RAM
than the machine has. Switched to streaming months through a `ParquetWriter`,
holding peak memory at roughly one month regardless of how many years are ingested.

**Null contamination.** Eleven months into a backfill, flattening crashed on a
report with `[None]` inside `product_problems`. `dict.get(key, "")` only protects
against a *missing* key — an explicit null still returns `None`.

**Quadratic dedup.** The fuzzy pass ran fine on synthetic test data and appeared
to hang on 4.6M rows: a pure-Python pairwise loop with pandas label lookups per
comparison. Batching through `rapidfuzz.process.cdist` brought it under 20 minutes.

**A dedup rate that was too good to be true.** The first clean run reported 49.8%
duplicates, well above published MAUDE research, so I went looking for my own bug.
First hypothesis — empty narratives hashing identically — tested and rejected
(only 0.9% are empty). The real cause was manufacturer template text: 251,854
separate dental implant reports all read "FAILURE TO OSSEOINTEGRATE." Distinct
events, identical wording. Added a frequency heuristic (text appearing more than
five times is boilerplate, not a duplicate) plus event date in the key. Corrected
rate: **5.1%**, consistent with the literature.

**A silent 100% failure rate.** The first LLM evaluation reported 1.0% accuracy on
failure mode but 82% on harm. That combination does not describe a bad model, so I
checked the output distribution: every row was the fallback value. A wrong model
name was 404ing all 300 requests while the `except` block swallowed it — the 82%
was the fallback happening to match the majority harm class. Added a pre-flight
model check, loud error reporting, and a circuit breaker that aborts if early
outputs look like fallbacks.

**Label drift in my own ground truth.** Four categories scored *identically* under
two unrelated classifiers, which points at the labels rather than the models.
`device_malfunction` had drifted into a catch-all during labelling — I had tagged
"the dental implant failed to osseointegrate" as `device_malfunction` when both
classifiers correctly said `implant_integration`. My own rule was "specific
mechanism wins" and I had stopped following it around hour three. Adjudicating
raised measured keyword accuracy from 38.7% to 47.3%. The classifier never
changed; only the measurement got honest.

**Classification errors propagate into signals.** Two of the top three emerging
alerts turned out to be taxonomy precedence bugs wearing the costume of safety
signals — recall returns classified as contamination, battery reports classified
as sensor accuracy. A classifier that is right about half the time produces alerts
that are wrong in structured, plausible-looking ways, which is an argument for
end-to-end investigation rather than trusting any single stage's metrics.

## Known limitations

- **Keyword rules were frozen before evaluation** and not tuned against the
  labelled set, because tuning on eval data inflates the number you report. Known
  gaps therefore stay unfixed — MAUDE writes "alert" where the taxonomy matches
  only "alarm" — and are listed as future work rather than quietly patched. (The
  precedence reordering was driven by signal-layer investigation, a different
  evidence source; accuracy was re-measured after.)
- **The `device_malfunction` vs `unknown` boundary is genuinely ambiguous** when a
  narrative names no mechanism. Single-annotator labels mean no inter-annotator
  agreement estimate — the main thing I would add next.
- **The LLM comparison is against a locally-served 3B model**, chosen because it is
  free and runs offline. This licenses no claim about LLMs generally.
- **The empirical-Bayes layer is a Gamma-Poisson model with a single
  method-of-moments prior** — a deliberate simplification of DuMouchel's MGPS
  mixture. Shrinkage still suppresses small-count false positives.
- **Some product codes are missing from the FDA classification table**, so a
  handful of alerts show panel "Unknown".
- **Artifact detection is heuristic.** The cross-device pass was added only after
  a manufacturer-wide filing change fragmented into five individually-clean
  device alerts; there are likely other patterns it still misses.
- **Retrospective recall validation is not yet run.** The code path exists and
  recomputes signals `as_of` each month so no future data leaks in.
- **MAUDE is passive surveillance.** Nothing here is evidence that any specific
  device is unsafe.

## Architecture

```
src/ingest.py         openFDA ingestion, recursive window splitting, parquet month caches
src/dedup.py          manufacturer normalization/clustering + template-aware dedup
src/extract.py        failure-mode & harm extraction (rules | llm) + labeling loop
src/device_meta.py    FDA classification lookup: device names + specialty panels
src/signals.py        self-controlled temporal alerting + disproportionality context
src/artifact_guard.py reporting-artifact detection, incl. cross-device filing changes
src/investigate.py    drill into one alert: narratives and manufacturers vs baseline
src/validate.py       retrospective check vs FDA recalls (as_of recomputation)
src/dashboard.py      Streamlit UI
run_pipeline.py       stage-by-stage orchestrator
eval_llm_ollama.py    local-LLM vs keyword evaluation against hand labels
tests/test_smoke.py   offline end-to-end check with a planted signal
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
python -m src.artifact_guard
streamlit run src/dashboard.py
```

Start with one quarter before widening. A multi-year backfill is several hours
and a few GB of cache.

## Statistics reference

Disproportionality (context tab) uses the standard 2×2 table per
(product_code × failure_mode) **within a medical specialty panel**: PRR, ROR,
Yates-corrected chi², and a Gamma-Poisson empirical-Bayes posterior. EB05 — the
5th percentile of that posterior — is the conservative bound real signal-detection
systems rank on. chi² is displayed but not used for flagging: at 4.4M reports
nearly everything is "significant," so it filters nothing.

Alerting (primary tab) computes, per device and failure mode, that mode's share of
the device's monthly reports, then a z-score against the device's own trailing
12-month baseline. Share rather than raw count, so a device's overall reporting
volume drifting over time neither creates nor masks an alert.

`signals.run(as_of=...)` recomputes signals using only data available at that
point in time, which is what would make retrospective recall validation honest
rather than circular.

## Data sources

- [openFDA Device Adverse Events](https://open.fda.gov/apis/device/event/) — MAUDE reports
- [openFDA Device Classification](https://open.fda.gov/apis/device/classification/) — device names and specialty panels
- [openFDA Device Recalls](https://open.fda.gov/apis/device/recall/) — validation ground truth
