# Task A1 — Constraint Analysis: Edge AI for Cold-Chain Truck Monitoring

## Scenario

LogiBridge monitors refrigerated pharmaceutical cargo trucks on the
Nashik–Aurangabad route (Maharashtra, India). Each truck carries a
temperature sensor (1 Hz) and a vibration sensor (500 Hz × 3 axes raw;
0.5 Hz RMS published for classification). The fault-to-alert budget is
**90 seconds**: a refrigeration failure raises cargo temperature at
~1 °C/min, so a 90 s delay risks a ~1.5 °C excursion — unacceptable for
vaccines and biologics with a ±2 °C tolerance band.

---

## 1. Latency

**Cloud round-trip on a rural Indian cellular link:**
A typical 4G RTT on the Nashik–Aurangabad highway is 80–200 ms under
good signal; during handoff or congestion it rises to 1–5 seconds.
Even at best-case 200 ms, a cloud-only pipeline adds:

```
sensor → MQTT publish → cellular uplink → cloud inference → downlink alert
≈ 200 ms RTT + broker queuing + inference + alert delivery
≈ 500 ms – 2 s under good conditions
```

This is technically within the 90 s budget *when connectivity exists*.
However, the route has **7 known blackout zones with 35–90 minute gaps**
(see Section 3). During a blackout the cloud pipeline produces zero
inferences — the truck is blind for up to 90 minutes, far exceeding the
90 s SLA.

**Edge architecture latency (this repository):**
The TFLite MLP inference on a Raspberry Pi 5 takes < 1 ms per window
(benchmark: M3 mean = 0.029 ms on x86; ARM with INT8 NEON is comparable).
The sliding-window step is 10 s, so the worst-case fault-to-alert latency
is **10 s** — 9× inside the 90 s budget, with zero dependency on
connectivity.

---

## 2. Bandwidth

**Raw sensor data rate per truck:**

| Stream | Rate | Payload | Bytes/day |
|---|---|---|---|
| Temperature | 1 Hz | 8 B float + 30 B JSON overhead | ~3.3 MB/day |
| Vibration (raw) | 500 Hz × 3 axes | 4 B × 3 per sample | ~518 MB/day |
| **Total raw** | | | **~521 MB/day** |

At ₹0.10/MB (typical Indian cellular data rate):

```
Raw streaming cost per truck per day = 521 MB × ₹0.10 = ₹52.10/day
85-truck fleet annual cost           = ₹52.10 × 85 × 365 ≈ ₹16.2 lakh/year
```

**Edge-processed output (this repository):**
The inference service publishes one small JSON result every 10 s:

```json
{"truck_id":"TRK-001","timestamp":"...","class_id":0,"class_label":"Normal","confidence":0.99}
```

~120 bytes × 8640 windows/day = ~1 MB/day per truck (all classes).
In practice only `Warning`/`Critical` results need uplink; under normal
operation that is < 50 messages/day ≈ **6 KB/day**.

```
Edge alert cost per truck per day ≈ 0.006 MB × ₹0.10 = ₹0.0006/day
85-truck fleet annual cost        ≈ ₹19/year  (vs ₹16.2 lakh for raw streaming)
```

**Bandwidth reduction: ~87,000×.**

---

## 3. Connectivity

**Known blackout locations:** 7 sites on the Nashik–Aurangabad NH-160
corridor, each causing 35–90 minute signal loss (tunnels, ghats, rural
dead zones).

**Cloud-only behaviour during a blackout:**
- The MQTT broker on the truck queues raw sensor payloads locally.
- No inference runs — the truck has no awareness of its own cargo state.
- If the refrigeration unit fails at the start of a 90-minute blackout,
  the cargo temperature rises ~90 °C (at 1 °C/min) before any alert
  fires. The entire batch is lost.
- On reconnection, the backlog floods the uplink, causing further delay.

**Edge architecture behaviour during a blackout:**
- The inference service (`inference_service.py`) runs entirely on-device
  over the local Mosquitto broker — no internet required.
- Classification continues uninterrupted at 10 s intervals throughout
  the blackout.
- Only the small inference-result payloads (~120 B each) are buffered for
  uplink sync on reconnection — negligible queue size even after 90 minutes
  (90 min × 6 msgs/min × 120 B ≈ 65 KB).
- A `Critical` alert triggers a local cab buzzer/light within 10 s of
  fault onset, regardless of connectivity.

---

## 4. Privacy

Raw cargo telemetry (temperature time-series, vibration waveforms,
GPS coordinates) constitutes commercially sensitive supply-chain data
and may be subject to pharmaceutical client data-handling contracts
(e.g. GDP guidelines, client NDAs).

In this architecture **raw sensor data never leaves the truck**. The only
payload transmitted to the cloud is the classification result:
`{class_label, confidence, truck_id, timestamp}` — four fields with no
raw sensor values and no location data. This design:

- Satisfies contractual obligations to pharmaceutical clients who prohibit
  raw telemetry leaving their custody chain.
- Reduces the attack surface: a compromised uplink channel exposes only
  `Normal/Warning/Critical` labels, not the underlying sensor stream.
- Simplifies DPDP Act 2023 (India) compliance — the uplink payload
  contains no personal or sensitive data.
