# Task A1 — Constraint Analysis (template)

Fill in each section below with your own reasoning (350-500 words total for the Final Report).
Use the reference numbers already given in the brief plus outputs from this codebase.

## Latency
- Fault-to-alert budget: 90 seconds. Failure signature raises cargo temperature at 1°C/min.
- State your assumed round-trip time for a rural Indian cellular link (research a typical 3G/4G
  RTT figure, or use a range e.g. 200 ms best case to several seconds during handoff/congestion)
  and argue whether a cloud round trip is safely inside 90 seconds even under good conditions,
  versus a connectivity gap where it is not possible at all.

## Bandwidth
- Temperature: 1 Hz. Vibration: 500 Hz x 3 axes (for this calculation only — the simulator itself
  publishes a single RMS value at 0.5 Hz for classification purposes; use the 500 Hz raw-sensor
  figure here because Task A1 asks about raw sensor bandwidth, not the classifier's input rate).
- Compute bytes/day per truck for raw streaming vs. the edge-processed alert volume (a handful of
  small JSON messages per day), then multiply by ₹0.10/MB for both cases.

## Connectivity
- Nashik-Aurangabad route: 35-90 minute blackouts at 7 known locations.
- Describe what a cloud-only architecture does during a blackout (queues locally with no
  inference, or fails open/closed) versus the edge architecture in this repository, which
  classifies locally and buffers only the small `inference` topic payloads for later sync.

## Privacy
- Explain that raw cargo/location telemetry never leaves the truck in this architecture; only
  classification results (Normal/Warning/Critical + confidence) are queued for uplink, which
  supports contractual data-handling commitments to pharmaceutical clients.
