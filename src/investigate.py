"""Investigate one alert: is it a real change in events, or a change in language?

Text-derived surveillance detects changes in how reports are WRITTEN as well as
changes in what HAPPENED. A manufacturer adopting a new narrative template can
reclassify tens of thousands of reports overnight without any device changing.
This tool shows what the narratives actually looked like before and during an
alert month so you can tell the two apart.

    python -m src.investigate --device "Pump, Infusion" --mode sensor_accuracy --month 2026-04
    python -m src.investigate --code FRN --mode software --month 2026-01

Read the output for:
- A narrative prefix that is ABSENT in the baseline and DOMINANT in the alert
  month  -> reporting artifact (new template), not a device change.
- The same narrative types as baseline, just more of them  -> plausibly real.
- One manufacturer appearing from nowhere -> check whether they changed filing
  practice, or whether this is a genuine single-manufacturer event.
"""

import argparse
from pathlib import Path

import pandas as pd

GOLD_DIR = Path("data/gold")
PREFIX = 90          # narrative prefix length used to group templates
BASELINE_MONTHS = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", help="device_name (as shown in the dashboard)")
    ap.add_argument("--code", help="product_code, if you prefer")
    ap.add_argument("--mode", required=True, help="failure_mode")
    ap.add_argument("--month", required=True, help="alert month, YYYY-MM")
    a = ap.parse_args()

    df = pd.read_parquet(GOLD_DIR / "events_extracted.parquet")
    if a.code:
        df = df[df["product_code"] == a.code]
        label = a.code
    else:
        from . import device_meta
        df = device_meta.annotate(df)
        df = df[df["device_name"] == a.device]
        label = a.device
    if df.empty:
        raise SystemExit(f"No reports found for {label}")

    months = sorted(df["month"].unique())
    idx = months.index(a.month) if a.month in months else None
    if idx is None:
        raise SystemExit(f"Month {a.month} not present. Available: {months[-6:]}")
    baseline_months = months[max(0, idx - BASELINE_MONTHS):idx]

    print(f"\n=== {label} | {a.mode} ===")

    # 1. Monthly volume, all modes vs this mode
    vol = (df.groupby("month")
             .agg(total=("report_key", "size"),
                  this_mode=("failure_mode", lambda s: (s == a.mode).sum())))
    vol["share"] = (vol["this_mode"] / vol["total"]).round(3)
    print("\nMonthly volume (last 12 months shown):")
    print(vol.tail(12).to_string())

    alert = df[(df["month"] == a.month) & (df["failure_mode"] == a.mode)]
    base = df[(df["month"].isin(baseline_months)) & (df["failure_mode"] == a.mode)]

    # 2. Narrative templates
    def templates(frame, n=6):
        if frame.empty:
            return pd.Series(dtype=int)
        return frame["narrative"].str.slice(0, PREFIX).value_counts().head(n)

    print(f"\nTop narrative openings in ALERT month ({a.month}, n={len(alert):,}):")
    for text, count in templates(alert).items():
        print(f"  {count:>7,}  {text}")

    print(f"\nTop narrative openings in BASELINE ({baseline_months[0]}..{baseline_months[-1]}, "
          f"n={len(base):,}):")
    for text, count in templates(base).items():
        print(f"  {count:>7,}  {text}")

    # 3. Is the dominant alert-month template new?
    if not alert.empty:
        top_text = templates(alert, 1).index[0]
        seen_before = base["narrative"].str.slice(0, PREFIX).eq(top_text).sum()
        print(f"\nDominant alert template appeared {seen_before:,} times in the "
              f"{len(baseline_months)}-month baseline.")
        if seen_before == 0:
            print("  --> NEW TEMPLATE. This alert is very likely a reporting "
                  "artifact, not a device change.")
        else:
            print("  --> Template existed before; volume change may be real.")

    # 4. Manufacturer mix
    print("\nManufacturers, alert month vs baseline:")
    am = alert["manufacturer"].value_counts().head(5)
    bm = base["manufacturer"].value_counts().head(5)
    print("  ALERT:   ", dict(am))
    print("  BASELINE:", dict(bm))


if __name__ == "__main__":
    main()