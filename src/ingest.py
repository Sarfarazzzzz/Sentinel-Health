"""Ingest FDA MAUDE device adverse-event reports via the openFDA API.

v3 - disk-efficient caching:
- v2 cached raw JSON per month (gigabytes/month) and filled the disk.
  v3 flattens events immediately and caches compact parquet per month
  under data/raw/flat/ (roughly 10-20x smaller). Raw JSON is never
  written to disk.
- Keeps v2's recursive window splitting (openFDA caps pagination at
  skip=25,000) and connection-error retries with backoff.
- PAGE_SIZE=1000 (openFDA's max limit) for 10x fewer requests.

Rename this file to ingest.py and place it in src/ (replacing the old one).
"""

import hashlib
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.fda.gov/device/event.json"
PAGE_SIZE = 1000  # openFDA max per request
SKIP_CAP = 25000  # openFDA hard cap on skip
FLAT_DIR = Path("data/raw/flat")
SILVER_DIR = Path("data/silver")


def _month_windows(start: str, end: str):
    months = pd.period_range(start=start, end=end, freq="M")
    for m in months:
        yield m.start_time.strftime("%Y%m%d"), m.end_time.strftime("%Y%m%d")


def _get(params: dict, max_retries: int = 6) -> dict:
    """GET with retries on both bad statuses and dropped connections."""
    key = os.environ.get("OPENFDA_API_KEY")
    if key:
        params = {**params, "api_key": key}
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=90)
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(min(2 ** attempt * 3, 60))
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return {"results": [], "meta": {"results": {"total": 0}}}
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** attempt * 5, 120))
            continue
        resp.raise_for_status()
    raise RuntimeError(f"openFDA kept failing for params={params}: {last_err}")


def _total_for(search: str) -> int:
    meta = _get({"search": search, "limit": 1})
    return int(meta.get("meta", {}).get("results", {}).get("total", 0))


def _page_through(search: str) -> list[dict]:
    out, skip = [], 0
    while skip <= SKIP_CAP:
        data = _get({"search": search, "limit": PAGE_SIZE, "skip": skip})
        results = data.get("results", [])
        out.extend(results)
        if len(results) < PAGE_SIZE:
            break
        skip += PAGE_SIZE
        time.sleep(0.25)
    return out


def _split(first_day: str, last_day: str):
    d0 = datetime.strptime(first_day, "%Y%m%d").date()
    d1 = datetime.strptime(last_day, "%Y%m%d").date()
    mid = d0 + (d1 - d0) / 2
    return ((d0.strftime("%Y%m%d"), mid.strftime("%Y%m%d")),
            ((mid + timedelta(days=1)).strftime("%Y%m%d"), d1.strftime("%Y%m%d")))


def fetch_range(first_day: str, last_day: str) -> list[dict]:
    """Fetch [first_day, last_day], splitting recursively past the 25k cap."""
    search = f"date_received:[{first_day} TO {last_day}]"
    total = _total_for(search)
    if total == 0:
        return []
    if total > SKIP_CAP:
        if first_day == last_day:
            print(f"  WARNING {first_day}: {total} reports in one day exceeds "
                  f"the {SKIP_CAP} cap; capturing the first {SKIP_CAP}.")
            return _page_through(search)
        print(f"  window {first_day}-{last_day}: {total:,} reports -> splitting")
        left, right = _split(first_day, last_day)
        return fetch_range(*left) + fetch_range(*right)
    return _page_through(search)


FLAT_COLUMNS = ["report_key", "date_received", "date_of_event", "event_type",
                "manufacturer_raw", "brand_name", "generic_name",
                "product_code", "product_problems", "narrative",
                "narrative_hash"]


def flatten(event: dict) -> dict:
    device = (event.get("device") or [{}])[0]
    texts = event.get("mdr_text") or []
    narrative = " ".join(
        (t.get("text") or "") for t in texts
        if (t.get("text_type_code") or "").lower().startswith("description")
    ) or " ".join((t.get("text") or "") for t in texts)
    return {
        "report_key": event.get("mdr_report_key"),
        "date_received": event.get("date_received"),
        "date_of_event": event.get("date_of_event"),
        "event_type": event.get("event_type"),
        "manufacturer_raw": device.get("manufacturer_d_name") or "",
        "brand_name": device.get("brand_name") or "",
        "generic_name": device.get("generic_name") or "",
        "product_code": device.get("device_report_product_code") or "",
        "product_problems": "; ".join(str(p) for p in (event.get("product_problems") or []) if p),
        "narrative": narrative.strip(),
        "narrative_hash": hashlib.md5(narrative.strip().lower().encode()).hexdigest(),
    }


def run(start: str, end: str) -> pd.DataFrame:
    """Ingest [start, end] (YYYY-MM-DD). Caches flattened parquet per month
    in data/raw/flat/YYYYMM.parquet; completed months are skipped on re-run."""
    FLAT_DIR.mkdir(parents=True, exist_ok=True)
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for first, last in _month_windows(start, end):
        month = first[:6]
        cache = FLAT_DIR / f"{month}.parquet"
        if cache.exists():
            df_m = pd.read_parquet(cache)
            print(f"  {month}: {len(df_m):,} reports (from cache)")
        else:
            events = fetch_range(first, last)
            df_m = (pd.DataFrame([flatten(e) for e in events])
                    if events else pd.DataFrame(columns=FLAT_COLUMNS))
            df_m.to_parquet(cache, index=False)
            print(f"  {month}: {len(df_m):,} reports "
                  f"({cache.stat().st_size / 1e6:.0f} MB cached)")
        frames.append(df_m)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df["month"] = pd.to_datetime(
            df["date_received"], format="%Y%m%d", errors="coerce"
        ).dt.to_period("M").astype(str)
        out = SILVER_DIR / "events.parquet"
        df.to_parquet(out, index=False)
        print(f"Silver table: {len(df):,} rows -> {out}")
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=str(date.today()))
    a = p.parse_args()
    run(a.start, a.end)
