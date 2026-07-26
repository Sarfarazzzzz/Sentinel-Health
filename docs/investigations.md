# Alert investigations

Raw output from `src/investigate.py` for alerts surfaced by the pipeline, with
the conclusion reached in each case. These are the adjudications that turned
"the system flagged something" into "here is what the something was."

Every alert investigated so far has traced to **reporting behaviour** rather than
device behaviour. That is the central finding of the project, not a failure of
it: MAUDE is passive surveillance, and passive surveillance data reflects how
events are reported at least as strongly as how often they occur.

---

## 1. Continuous glucose monitor — `software`, 2026-01

**Verdict: batch retrospective filing of a real issue.**

The dominant narrative template existed before the spike (1,524 occurrences over
the six-month baseline, roughly 250/month) and then appeared 54,724 times in a
single month. The device's *total* report volume also tripled that month (83,610
vs ~30k typical), which is the fingerprint of batch filing rather than a change
in event rate.

The underlying issue is real — a G7 coding defect where a failed sensor does not
trigger the expected failure alert — but those 54,724 events did not *happen* in
January 2026, they were *filed* in January 2026.

```
Monthly volume:
2025-11  22632   1059  0.047
2025-12  27370   2153  0.079
2026-01  83610  57279  0.685      <- alert month
2026-02  23134    736  0.032
2026-03  28374   1900  0.067

Top narrative opening, alert month:  54,724x  "DEXCOM BECAME AWARE THAT THE USER
                                                EXPERIENCED A FAILURE BUT DID NOT
                                                RECEIVE THE EXPECTED S..."
Same opening across the 6-month baseline: 1,524x
Manufacturers, alert month:  dexcom 57,261 | abbott diabetes care 17
```

---

## 2. Infusion pump — `sensor_accuracy`, 2026-04

**Verdict: reporting artifact *and* a classification error.**

New template with zero baseline occurrences, and one manufacturer (Fresenius
Kabi) supplied 857 of 870 reports having been absent from the baseline top five.

Reading the narratives also exposed a taxonomy bug: they describe *battery health*
assessments, but were classified `sensor_accuracy` because that category sat
above `battery_power` in the keyword precedence order, so a stray "discrepan"
matched before any battery keyword was tested. Fixed by reordering (see README).

```
Top narrative opening, alert month:  522x  "THE FOLLOWING HAS BEEN REPORTED: AN
                                             ASSESSMENT OF A CUSTOMER BATTERY
                                             HEALTH REPORT IDENTIF..."
Same opening across baseline: 0x
Manufacturers, alert month:  fresenius kabi 857 | carefusion sd 4 | baxter 2
Manufacturers, baseline:     carefusion sd 45 | icu medical 34 | baxter 15
```

---

## 3. Ventilator — `contamination`, 2026-02

**Verdict: recall remediation campaign, misclassified.**

The narratives are BiPAP devices returned to third-party service centers under a
voluntary field safety notice — the Philips Respironics sound-abatement foam
recall. These are recall-remediation returns and should have been
`recall_field_action`, but `contamination` sat higher in the keyword precedence
order so "particles in device" matched first.

Two fixes followed: `recall_field_action` moved to the top of the taxonomy
(a report *type* takes precedence over whatever failure language it contains),
and recall-remediation reports were excluded from alerting entirely — a spike in
them means a recall campaign is running, which is the opposite of an early
warning.

```
Top narrative opening, alert month:  41x  "A BIPAP A40 DEVICE WAS RETURNED TO A
                                            THIRD-PARTY SERVICE CENTER FOR SERVICE
                                            AS PART OF THE..."
Same opening across baseline: 0x
Manufacturers:  respironics 214 (alert) | respironics 31 (baseline)
```

---

## 4. Endoscopes — `leak_seal`, 2026-04/05 (five device types)

**Verdict: coordinated manufacturer filing change across a product line.**

This one looked like the best signal the system had produced: five *different*
endoscope types — duodenoscope, ureteroscope, colonoscope, gastroscope,
bronchoscope — from nominally different product codes, all spiking on the same
failure mode in the same two months. Endoscope leak integrity is a genuine safety
concern (a breach lets fluid into the channel, which is a contamination pathway),
so a cross-device signal there would have been meaningful.

Investigation showed Olympus subsidiaries accounted for **100%** of both spikes
checked, with new narrative templates in each. Same corporate parent, same month,
same new phrasing across separate product lines.

The events are probably real — scopes genuinely failing leak tests during
reprocessing. What changed is that Olympus began routing them to MAUDE
systematically.

```
Duodenoscope, 2026-05:  32 reports
  dominant template:  "IT WAS OBSERVED THAT DURING THE DEVICE EVALUATION, THE
                       DUODENOVIDEOSCOPE EXHIBITED DAMAGED..."   (0x in baseline)
  manufacturers:  aizu olympus 32 / 32

Ureteroscope, 2026-05:  67 reports
  dominant template:  "THE URETERO-RENO VIDEOSCOPE WAS RETURNED TO THE SERVICE
                       CENTER WITH A REPORT OF FAILED LEA..."    (0x in baseline)
  manufacturers:  aizu olympus 59 | shirakawa olympus 8   (67 / 67 Olympus)
```

**This exposed a gap in the artifact guard.** The per-device checks each passed:
Aizu Olympus was *already* the baseline leader for every scope type, so no
"manufacturer shift" was detectable within any single device, and the new-template
check evaluated one product code at a time. The coordinated timing across product
codes was the only visible giveaway.

Fixed by adding a cross-device pass: if one manufacturer shows new narrative
templates across three or more product codes in the same month, all of those
alerts are annotated as a coordinated filing change.
