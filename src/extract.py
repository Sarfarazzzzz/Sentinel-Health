"""Extract structured failure signals from free-text MAUDE narratives.

Two modes:
- rules  : keyword taxonomy. Free, instant, and a legitimate baseline.
- llm    : Anthropic API (set ANTHROPIC_API_KEY). Extracts the same schema
           zero-shot. Costs money; run it on a sample first.

The evaluation loop is the part that matters for your portfolio:
1. `python -m src.extract --make-labels 300` writes a CSV sample.
2. YOU hand-label the failure_mode and harm columns in that CSV.
3. `python -m src.extract --evaluate labels.csv` scores both modes against
   your labels. The measured comparison (and your decision about which mode
   to ship) is the story you tell in interviews.
"""

import json
import os
from pathlib import Path

import pandas as pd

SILVER_DIR = Path("data/silver")
GOLD_DIR = Path("data/gold")

# Starter taxonomy — extend it as you read real narratives. Reading a few
# hundred narratives and growing this map IS the domain work.
FAILURE_KEYWORDS = {
    # --- report TYPE first: takes precedence over failure language ---
    "recall_field_action": ["field action", "field safety notice",
                            "recall notification", "known issue",
                            "correction and removal", "recall no",
                            "voluntary field safety", "recall remediation",
                            "returned to a third-party service center"],
    # --- highly specific device/clinical events ---
    "implant_integration": ["osseointegrat", "primary stability", "implant mobility",
                            "failure upon insertion", "covered with bone",
                            "dental implant loss", "lack of stability", "loosening",
                            "metallosis"],
    "battery_power": ["battery", "power loss", "powered off", "depleted",
                      "recharge", "charging", "battery charge", "power source",
                      "low voltage", "rejects new batteries"],
    "sensor_accuracy": ["inaccura", "erroneous", "error grid", "low glucose result",
                        "high glucose result", "high reading", "low reading",
                        "points off", "compared to readings", "discrepan",
                        "difference between sensor", "sensor glucose vs",
                        "sensor deviation", "sensor error"],
    "glycemic_event": ["hyperglycemia", "hypoglycemia", "blood glucose level rose",
                       "elevated blood glucose", "bg level rose", "rose to",
                       "diabetic coma", "fluctuating blood glucose"],
    "explant_revision": ["explant", "revised due to", "revision surgery",
                         "implant removed", "removed due to", "reason for removal",
                         "nonfunctioning", "underwent a revision"],
    "contamination": ["contaminat", "foreign material", "foreign matter", "debris",
                      "mycobacteria", "particles in device", "particulate", "mold"],
    "early_failure": ["early sensor expiration", "expired early", "premature",
                      "did not last", "doesn't last", "wear period"],
    "dosing_delivery": ["overinfusion", "underinfusion", "over-delivery", "bolus",
                        "occlusion", "flow blocked", "bent cannula", "infusion set",
                        "cartridge alarm", "dose", "insulin flow", "pod fell off",
                        "cannula dislodge", "delayed deployment",
                        "partially inserted", "reduced concentration"],
    "imaging_quality": ["noisy image", "image interference", "image quality",
                        "blurry", "scratches on tip", "artifact", "no image"],
    "cardiac_lead": ["pacing impedance", "lead sensing", "sensing value",
                     "capture threshold", "lead explanted", "lead fracture",
                     "shock impedance", "undersensing", "oversensing",
                     "atrial channel", "retracting its helix", "helix"],
    # --- generic mechanisms ---
    "software": ["software", "firmware", "error code", "froze", "reboot",
                 "crash", "unresponsive", "black screen", "fault",
                 "coding issue", "failed to boot", "unable to boot",
                 "failed to start", "inoperative", "interface issue",
                 "data was not current", "forced log out"],
    "mechanical_break": ["fracture", "fractured", "broke", "broken", "crack",
                         "detach", "sheared", "hole", "exposed electronics",
                         "stuck", "sticking", "jammed", "would not open",
                         "did not recoil", "failed to deploy", "failed to pass",
                         "failed to advance", "needle mechanism", "was missing",
                         "separated", "comes loose", "came loose", "kink",
                         "frayed", "crushed", "malformed", "peeling",
                         "physically damaged", "could not fire",
                         "did not return"],
    "leak_seal": ["leak", "leaking", "seal", "rupture", "burst",
                  "water tightness"],
    "electrical": ["short circuit", "sparking", "overheat", "burned", "smoke"],
    "alarm": ["alarm failed", "no alarm", "alarm did not", "failed to alert"],
    "connectivity": ["disconnect", "bluetooth", "signal loss",
                     "loss of communication", "no communication", "telemetry",
                     "transmission", "does not communicate", "not connected"],
    "biocompatibility": ["infection", "inflammation", "allergic", "reaction",
                         "migration", "hemolysis", "hyperplasia", "thrombosis",
                         "thrombus", "emboli"],
    # --- deliberate catch-all, must stay LAST ---
    "device_malfunction": ["malfunction", "critical pump error", "pump error",
                           "device failure", "failed pm", "motor error",
                           "error alarm", "alarm occurred", "drawer fail",
                           "kept failing"],
}
HARM_KEYWORDS = {
    "death": ["death", "died", "fatal", "expired"],
    "serious_injury": ["hospitalized", "surgery", "life threatening", "permanent", "amputat"],
    "injury": ["injury", "injured", "harm", "burn", "shock", "wound"],
}

LLM_PROMPT = """You are extracting structured data from an FDA medical device adverse event narrative.

Narrative:
\"\"\"{narrative}\"\"\"

Respond with ONLY a JSON object, no other text, with keys:
- failure_mode: one of {modes} or "other" or "unknown"
- harm: one of ["death", "serious_injury", "injury", "malfunction_only", "unknown"]
- component: short free text naming the implicated component, or null
- device_in_use: true if the device was in active use during the event, else false or null
"""

def classify_rules(narrative: str, event_type: str) -> dict:
    if not (narrative or "").strip():
        return {"failure_mode": "no_narrative", "harm": "unknown"}
    text = (narrative or "").lower()
    failure_mode = "unknown"
    for mode, kws in FAILURE_KEYWORDS.items():
        if any(k in text for k in kws):
            failure_mode = mode
            break
    harm = "malfunction_only"
    for level, kws in HARM_KEYWORDS.items():
        if any(k in text for k in kws):
            harm = level
            break
    # event_type from the structured record trumps keywords when explicit.
    et = (event_type or "").lower()
    if "death" in et:
        harm = "death"
    elif "injury" in et and harm == "malfunction_only":
        harm = "injury"
    return {"failure_mode": failure_mode, "harm": harm}


def classify_llm(narrative: str, model: str = "claude-haiku-4-5-20251001") -> dict:
    """Zero-shot extraction with the Anthropic API. pip install anthropic."""
    import anthropic
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    msg = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": LLM_PROMPT.format(
            narrative=narrative[:4000], modes=list(FAILURE_KEYWORDS))}],
    )
    raw = msg.content[0].text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"failure_mode": "unknown", "harm": "unknown"}


def run(mode: str = "rules", sample: int | None = None) -> pd.DataFrame:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SILVER_DIR / "events_dedup.parquet")
    if sample:
        df = df.sample(sample, random_state=7)
    if mode == "rules":
        extracted = [classify_rules(n, e) for n, e in zip(df["narrative"], df["event_type"])]
    else:
        extracted = [classify_llm(n) for n in df["narrative"]]
    ex = pd.DataFrame(extracted, index=df.index)
    df = pd.concat([df, ex[["failure_mode", "harm"]]], axis=1)
    out = GOLD_DIR / "events_extracted.parquet"
    df.to_parquet(out, index=False)
    print(df["failure_mode"].value_counts().to_string())
    return df


def make_label_sample(n: int = 300):
    df = pd.read_parquet(SILVER_DIR / "events_dedup.parquet")
    pool = df[df["narrative"].str.len() > 80]
    sample = pool.sample(min(n, len(pool)), random_state=7)
    cols = ["report_key", "narrative"]
    sample = sample[cols].assign(failure_mode="", harm="")
    path = Path("data/labels_todo.csv")
    sample.to_csv(path, index=False)
    print(f"Wrote {len(sample)} narratives to {path}. Label the empty columns by hand.")


def evaluate(labels_csv: str):
    labels = pd.read_csv(labels_csv, dtype={"report_key": str})
    labels = labels[labels["failure_mode"].astype(str).str.len() > 0]
    preds = [classify_rules(n, "") for n in labels["narrative"]]
    pred_df = pd.DataFrame(preds)
    for col in ["failure_mode", "harm"]:
        acc = (pred_df[col].values == labels[col].values).mean()
        print(f"rules-mode accuracy on {col}: {acc:.2%}  (n={len(labels)})")
    print("Run the same comparison with --mode llm on the labeled subset, "
          "then write up which you'd ship and why. That paragraph is the deliverable.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["rules", "llm"], default="rules")
    p.add_argument("--sample", type=int)
    p.add_argument("--make-labels", type=int, metavar="N")
    p.add_argument("--evaluate", metavar="LABELS_CSV")
    a = p.parse_args()
    if a.make_labels:
        make_label_sample(a.make_labels)
    elif a.evaluate:
        evaluate(a.evaluate)
    else:
        run(a.mode, a.sample)
