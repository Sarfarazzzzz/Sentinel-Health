"""Does the cross-sectional (disproportionality) flag predict recalls?

The temporal alert caught 1 of 40 recalls, because it detects CHANGE and recalls
typically follow chronic problems accumulating at a stable rate. This tests the
other detector: was the recalled device already flagged as disproportionate for
some failure mode, using ONLY data available before the recall?

Recomputes signals as_of the month before each recall (no lookahead). Results
are cached per month, since many recalls share a month.

    python check_crosssectional.py
"""

import pandas as pd

from src import signals

MIN_HISTORY_MONTHS = 6      # skip recalls too early for a meaningful baseline

val = pd.read_parquet("data/gold/validation.parquet")
events = pd.read_parquet("data/gold/events_extracted.parquet")
first_month = events["month"].min()
print(f"data starts {first_month}; evaluating recalls at least "
      f"{MIN_HISTORY_MONTHS} months after that")

cache: dict[str, set] = {}
rows = []

for _, rc in val.iterrows():
    as_of = str(pd.Period(rc["recall_month"], freq="M") - 1)
    if (pd.Period(as_of, freq="M") - pd.Period(first_month, freq="M")).n < MIN_HISTORY_MONTHS:
        print(f"  {rc['product_code']} {rc['recall_month']}: skipped (too early)")
        continue

    if as_of not in cache:
        ct = signals.run(as_of=as_of)
        if ct is None or ct.empty or "disproportionate" not in ct.columns:
            print(f"  (no signals computable as of {as_of})")
            cache[as_of] = set()
        else:
            cache[as_of] = set(ct.loc[ct["disproportionate"], "product_code"])
        print(f"  computed as_of {as_of}: {len(cache[as_of])} flagged devices")

    flagged = rc["product_code"] in cache[as_of]
    rows.append({"product_code": rc["product_code"],
                 "recall_month": rc["recall_month"],
                 "as_of": as_of,
                 "flagged_before": flagged})
    print(f"  {rc['product_code']} {rc['recall_month']}: {'FLAG' if flagged else '-'}")

out = pd.DataFrame(rows)
if out.empty:
    raise SystemExit("No recalls had enough prior history to evaluate.")

out.to_parquet("data/gold/validation_crosssectional.parquet", index=False)

# Base rate: what share of ALL devices are flagged, using the same as_of months?
base_rates = []
for as_of, flagged_set in cache.items():
    ct = signals.run(as_of=as_of)
    if ct is not None and not ct.empty and "product_code" in ct.columns:
        base_rates.append(len(flagged_set) / ct["product_code"].nunique())
base = sum(base_rates) / len(base_rates) if base_rates else float("nan")

print("\n=== Cross-sectional recall prediction (no lookahead) ===")
print(f"Recalls evaluated:            {len(out)}")
print(f"Flagged before recall:        {out['flagged_before'].sum()} "
      f"({out['flagged_before'].mean():.0%})")
print(f"Base rate across all devices: {base:.0%}")
print(f"Lift:                         {out['flagged_before'].mean() / base:.1f}x"
      if base and base == base else "")