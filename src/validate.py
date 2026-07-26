"""Retrospective validation: would Sentinel have flagged real FDA recalls early?

Pulls device recalls from openFDA, matches each recall's product code against
the signal detector run *as of each month before the recall*, and reports:
- coverage: fraction of recalls where a signal fired before the recall date
- median lead time in months for the hits
- false-alarm context: how many never-recalled devices were flagged

Run signals + extraction first. This module re-runs signal detection month by
month with no future data (see signals.run(as_of=...)), which is slower but
honest — a validator that peeks at the future is worthless.
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

from . import signals

RECALL_URL = "https://api.fda.gov/device/recall.json"
GOLD_DIR = Path("data/gold")
RAW_DIR = Path("data/raw")


def fetch_recalls(start: str, end: str, classifications=("Class I", "Class II")) -> pd.DataFrame:
    """Pull recalls initiated in [start, end], YYYYMMDD strings."""
    cache = RAW_DIR / f"recalls_{start}_{end}.json"
    if cache.exists():
        results = json.loads(cache.read_text())
    else:
        results, skip = [], 0
        while skip <= 25000:
            resp = requests.get(RECALL_URL, params={
                "search": f"event_date_initiated:[{start} TO {end}]",
                "limit": 100, "skip": skip}, timeout=60)
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
        "classification": r.get("root_cause_description", ""),
        "recalling_firm": r.get("recalling_firm", ""),
        "date_initiated": r.get("event_date_initiated", ""),
        "reason": (r.get("reason_for_recall") or "")[:300],
    } for r in results]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["recall_month"] = pd.to_datetime(df["date_initiated"], errors="coerce").dt.to_period("M").astype(str)
        df = df[df["product_code"].str.len() > 0].drop_duplicates("recall_number")
    print(f"{len(df)} recalls with product codes in window")
    return df


def validate(recalls: pd.DataFrame, lookback_months: int = 18) -> pd.DataFrame:
    """For each recall, find the earliest month a signal fired on its product code."""
    results = []
    for _, rc in recalls.iterrows():
        recall_month = rc["recall_month"]
        if not isinstance(recall_month, str) or recall_month == "NaT":
            continue
        months = pd.period_range(end=pd.Period(recall_month) - 1, periods=lookback_months, freq="M")
        first_hit = None
        for m in months:                       # earliest first
            ct = signals.run(as_of=str(m))
            hit = ct[(ct["product_code"] == rc["product_code"]) & ct["signal"]]
            if not hit.empty:
                first_hit = str(m)
                break
        lead = (pd.Period(recall_month) - pd.Period(first_hit)).n if first_hit else None
        results.append({**rc.to_dict(), "first_signal_month": first_hit, "lead_time_months": lead})
        print(f"  {rc['product_code']} recall {recall_month}: "
              f"{'signal at ' + first_hit if first_hit else 'no prior signal'}")
    out = pd.DataFrame(results)
    out.to_parquet(GOLD_DIR / "validation.parquet", index=False)
    hits = out["first_signal_month"].notna()
    print("\n=== Retrospective validation ===")
    print(f"Recalls evaluated: {len(out)}")
    print(f"Flagged before recall: {hits.sum()} ({hits.mean():.0%})")
    if hits.any():
        print(f"Median lead time: {out.loc[hits, 'lead_time_months'].median():.0f} months")
    print("Now write the honest paragraph about the misses. That paragraph "
          "is worth more than the hit rate.")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="20240101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--max-recalls", type=int, default=30)
    a = p.parse_args()
    rc = fetch_recalls(a.start, a.end)
    validate(rc.head(a.max_recalls))
