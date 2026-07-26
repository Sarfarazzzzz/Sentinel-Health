"""End-to-end pipeline runner for Sentinel-Health.

Usage:
    python run_pipeline.py --start 2023-01-01 --end 2026-06-30
    python run_pipeline.py --stages signals,validate
"""

import argparse
from datetime import date

from src import ingest, dedup, extract, signals, validate

STAGES = ["ingest", "dedup", "extract", "signals", "validate"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--stages", default=",".join(STAGES),
                   help="comma-separated subset of: " + ",".join(STAGES))
    p.add_argument("--extract-mode", choices=["rules", "llm"], default="rules")
    a = p.parse_args()
    todo = [s.strip() for s in a.stages.split(",")]

    if "ingest" in todo:
        print("\n=== 1/5 Ingest (openFDA MAUDE) ===")
        ingest.run(a.start, a.end)
    if "dedup" in todo:
        print("\n=== 2/5 Dedup + manufacturer normalization ===")
        dedup.run()
    if "extract" in todo:
        print("\n=== 3/5 Narrative extraction ===")
        extract.run(mode=a.extract_mode)
    if "signals" in todo:
        print("\n=== 4/5 Disproportionality signals ===")
        signals.run()
    if "validate" in todo:
        print("\n=== 5/5 Retrospective recall validation ===")
        rc = validate.fetch_recalls(a.start.replace("-", ""), a.end.replace("-", ""))
        validate.validate(rc.head(30))
    print("\nDone. Dashboard: streamlit run src/dashboard.py")


if __name__ == "__main__":
    main()
