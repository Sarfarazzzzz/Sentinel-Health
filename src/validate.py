"""Retrospective validation: would Sentinel have flagged real FDA recalls early?

Rewritten for the v4 alerting design. Two changes from the original:

1. It validates the TEMPORAL alert (trend z-score vs the device's own baseline),
   not the cross-sectional disproportionality flag, which is no longer the alert.
2. It computes the full monthly alert history ONCE and then reads it, instead of
   recomputing every signal table per recall per month. The as_of guarantee is
   preserved because the z-score baseline is a trailing window: the alert for
   month M is a function of months M-12..M only, so no future data can influence
   it. That is what makes reading a precomputed history equivalent to - and
   hundreds of times faster than - recomputing as_of each month.

    python -m src.validate --start 20240101 --end 20260630 --max-recalls 40
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from . import signals

RECALL_URL = "https://api.fda.gov/device/recall.json"
GOLD_DIR = Path("data/gold")
RAW_DIR = Path("data/raw")
LOOKBACK_MONTHS = 18


def fetch_recalls(start: str, end: str) -> pd.DataFrame:
    """Recalls initiated in [start, end] (YYYYMMDD)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"recalls_{start}_{end}.json"
    if cache.exists():
        results = json.loads(cache.read_text())
    else:
        results, skip = [], 0
        while skip <= 25000:
            resp = requests.get(RECALL_URL, params={
                "search": f"event_date_initiated:[{start} TO {end}]",
                "limit": 100, "skip": skip}, timeout=90)
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            page = resp.json().get("results", [])
            results.extend(page)
            if len(page) < 100:
                break
            skip += 100
            time.sleep(0.3)
        cache.write_text(json.dumps(results))

    rows = [{
        "recall_number": r.get("cfres_id") or r.get("res_event_number"),
        "product_code": r.get("product_code", ""),
        "recalling_firm": r.get("recalling_firm", ""),
        "date_initiated": r.get("event_date_initiated", ""),
        "reason": (r.get("reason_for_recall") or "")[:200],
    } for r in results]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["recall_month"] = (pd.to_datetime(df["date_initiated"], errors="coerce")
                            .dt.to_period("M").astype(str))
    df = df[df["product_code"].str.len() > 0]
    # One row per (device, month): a single recall event often produces many
    # records, and counting them separately would inflate the denominator.
    df = df.drop_duplicates(["product_code", "recall_month"])
    return df[df["recall_month"] != "NaT"]


def validate(recalls: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    """For each recall, find the earliest prior month an alert fired."""
    alerts = trends[trends["alert"]]
    results = []
    for _, rc in recalls.iterrows():
        window_end = pd.Period(rc["recall_month"], freq="M") - 1
        window_start = window_end - (LOOKBACK_MONTHS - 1)
        hits = alerts[(alerts["product_code"] == rc["product_code"])
                      & (alerts["month"] >= str(window_start))
                      & (alerts["month"] <= str(window_end))]
        if hits.empty:
            results.append({**rc.to_dict(), "first_signal_month": None,
                            "signal_failure_mode": None, "lead_time_months": None})
            continue
        first = hits.sort_values("month").iloc[0]
        lead = (pd.Period(rc["recall_month"], freq="M")
                - pd.Period(first["month"], freq="M")).n
        results.append({**rc.to_dict(),
                        "first_signal_month": first["month"],
                        "signal_failure_mode": first["failure_mode"],
                        "lead_time_months": lead})

    out = pd.DataFrame(results)
    out.to_parquet(GOLD_DIR / "validation.parquet", index=False)

    hits = out["first_signal_month"].notna()
    print("\n=== Retrospective recall validation ===")
    print(f"Recalls evaluated:      {len(out)}")
    print(f"Flagged before recall:  {hits.sum()} ({hits.mean():.0%})")
    if hits.any():
        lt = out.loc[hits, "lead_time_months"]
        print(f"Median lead time:       {lt.median():.0f} months "
              f"(range {lt.min():.0f}-{lt.max():.0f})")
        print("\nExamples:")
        print(out[hits].sort_values("lead_time_months", ascending=False)[
            ["product_code", "recall_month", "first_signal_month",
             "signal_failure_mode", "lead_time_months"]].head(10).to_string(index=False))
    print("\nNow read the misses and explain them. That paragraph is the deliverable.")
    return out


def run(start: str, end: str, max_recalls: int | None = None) -> pd.DataFrame:
    print("Computing full monthly alert history...")
    signals.run()                      # writes trends.parquet
    trends = pd.read_parquet(GOLD_DIR / "trends.parquet")
    print(f"  {int(trends['alert'].sum()):,} alert-months across all history")

    recalls = fetch_recalls(start, end)
    print(f"{len(recalls)} distinct device-recall events in window")

    # Only recalls for devices we actually have alert history for; otherwise the
    # denominator is padded with devices the system could never have flagged.
    known = set(trends["product_code"].unique())
    recalls = recalls[recalls["product_code"].isin(known)]
    print(f"{len(recalls)} of those are for devices present in the report data")

    if max_recalls:
        recalls = recalls.head(max_recalls)
    return validate(recalls, trends)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="20240101")
    p.add_argument("--end", default="20260630")
    p.add_argument("--max-recalls", type=int, default=40)
    a = p.parse_args()
    run(a.start, a.end, a.max_recalls)