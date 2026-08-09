# LogiEdge — Phase 1 Report
## System Architecture, Hardware Justification, and Deployment Context

**AIML ZG535 — Machine Learning on Edge | BITS Pilani WILP**
**Mini-Project Assignment 1 — Phase 1 Deliverable**

---

## Component A — System Architecture and Deployment Justification

### A1 — Constraint Analysis

#### 1. Latency Constraint

A refrigeration unit failure raises cargo temperature at approximately 1 deg C per minute.
The system SLA requires anomaly detection and alerting within **90 seconds** of a fault
signature appearing in sensor data.

**Cloud inference round-trip latency analysis (India rural cellular):**

| Segment | Latency |
|---|---|
| 4G uplink — rural Maharashtra | 80–200 ms |
| Cloud inference — AWS Mumbai region | 15–40 ms |
| 4G downlink alert delivery | 80–200 ms |
| **Total RTT — best case** | **175 ms** |
| **Total RTT — congestion / handoff** | **800 ms – 3 s** |

While a single RTT appears within budget, the 90-second SLA requires **continuous
window-based inference every 10 seconds**. During cellular handoffs — which occur at
7 known dead zones on the Nashik–Aurangabad route — cloud inference becomes unavailable
for 35–90 minutes per trip. A cloud-only system produces **zero alerts** during these
gaps, directly violating the SLA.

**Edge inference latency (this repository):**
TFLite INT8 MLP benchmark results:

| Model | Mean Latency | p95 Latency |
|---|---|---|
| M1 FP32 | 0.0078 ms | 0.0087 ms |
| M2 PTQ INT8 | 0.0269 ms | 0.0513 ms |
| M3 Pruned INT8 | 0.0288 ms | 0.0354 ms |

All three variants are well within the 90-second SLA. Edge inference is mandatory.

---

#### 2. Bandwidth Constraint

**Raw data generation per truck per day:**

| Stream | Rate | Payload | Daily Volume |
|---|---|---|---|
| Temperature | 1 Hz | ~38 B JSON | ~3.3 MB |
| Vibration (raw 500 Hz x 3 axes) | 500 Hz | 4 B x 3 per sample | ~518 MB |
| Door events | ~20/day | ~64 B each | ~1.3 KB |
| **Total raw** | | | **~521 MB/day** |

**Edge-processed output (inference results only):**

| Output | Frequency | Payload | Daily Volume |
|---|---|---|---|
| Inference result | Every 10 s | ~120 B JSON | ~1 MB |
| Alerts (Warning/Critical only) | ~5–10/day | ~120 B | ~1.2 KB |
| **Total edge output** | | | **~1 MB/day** |

**Cost comparison at Rs 0.10/MB:**

| Approach | Daily Data/Truck | Daily Cost/Truck | 85-Truck Fleet/Year |
|---|---|---|---|
| Raw cloud streaming | 521 MB | Rs 52.10 | Rs 16.2 lakh |
| Edge-processed alerts | ~1 MB | Rs 0.10 | Rs 31,025 |
| **Savings** | **99.8% reduction** | **Rs 52.00/truck/day** | **Rs 15.9 lakh/year** |

---

#### 3. Connectivity Constraint

**Connectivity profile — Nashik–Aurangabad route (NH-160):**
- 7 documented dead zones: Igatpuri ghat, Kasara stretch, rural patches near Sinnar,
  and four additional locations
- Estimated offline duration: **35–90 minutes per trip**

**Cloud-only system during a blackout:**
- All inference stops — no anomaly detection for the entire gap duration
- A refrigeration failure at gap entry goes undetected for up to 90 minutes
- At 1 deg C/min temperature rise, cargo could reach 90 deg C above setpoint before any alert fires
- Buffered sensor data arrives at cloud in a burst on reconnection — too late for intervention

**LogiEdge edge architecture during a blackout:**
- Inference service runs entirely over the local Mosquitto broker — no internet required
- Classification continues uninterrupted at 10 s intervals throughout the blackout
- Only small inference-result payloads (~120 B each) are buffered for uplink sync
- Maximum local buffer: 90 min x 6 windows/min x 120 B = **65 KB** — negligible

---

#### 4. Privacy Constraint

Pharmaceutical clients operate under Schedule M of the Drugs and Cosmetics Act and
WHO GDP guidelines, which require cargo telemetry to remain within the vehicle network.

**How on-device inference addresses these requirements:**

| Requirement | Edge Solution |
|---|---|
| No raw data to cloud | Only inference labels (Normal/Warning/Critical) and confidence scores transmitted |
| Local audit trail | Inference results logged locally with timestamps |
| DPA compliance | No raw pharmaceutical telemetry leaves the truck |
| DPDP Act 2023 (India) | Uplink payload contains no personal or sensitive data |

---

### A2 — System Architecture

```
Sensor Simulator (temperature, vibration)
            |  MQTT (localhost:1883)
            v
      Mosquitto Broker
            |
            v
   Inference Service
   (sliding window -> normalise -> TFLite model)
            |  MQTT (publishes classification result)
            v
   +--------+---------+
   |                  |
Console output    Drift Monitor (PSI vs reference distribution)
(OK / ALERT)      (flags [LOGIBRIDGE DRIFT ALERT] on PSI > 0.25)
```

**Classes predicted:** Normal (0), Warning (1), Critical (2)

---

## Component B — Hardware Selection and Justification

### B1 — Constraint Triangle Application

| Criterion | RPi 5 + AI HAT+ (13 TOPS) | Jetson Orin Nano (67 TOPS) | STM32H7 MCU |
|---|---|---|---|
| **Peak Power** | 7.5 W — within 10 W budget | 15 W — exceeds budget | 0.4 W |
| **Compute** | 13 TOPS — ample for 45 MFLOP MLP | 67 TOPS — severe overkill | Sufficient for MLP only |
| **Unit Cost** | ~Rs 15,000 | ~Rs 45,000 | ~Rs 3,500 |
| **Fleet Cost (85 trucks)** | Rs 12.75 lakh | Rs 38.25 lakh | Rs 2.975 lakh |
| **Fleet Cost (265 trucks)** | Rs 39.75 lakh | Rs 119.25 lakh | Rs 9.275 lakh |
| **Docker / Ansible OTA** | Yes | Yes | No |
| **90 s SLA** | Yes | Yes | No (no OS, no TFLite) |
| **Verdict** | **Selected** | Fails Power + Cost | Fails OTA + Performance |

**Decision: Raspberry Pi 5 + AI HAT+**

**Against Jetson Orin Nano:** 67 TOPS is designed for real-time video inference.
Running a 45 MFLOP MLP on it is architectural overkill. At 15 W it exceeds the 10 W
truck auxiliary power budget. The Rs 25.5 lakh premium over RPi 5 for the 85-truck
pilot is unjustifiable for this workload.

**Against STM32H7:** Bare-metal firmware with no OS, no Docker, no Python TFLite
runtime. The OTA update mechanism in this repository (Docker layer push via Ansible)
cannot run on an MCU. A custom bootloader would need to be built from scratch,
erasing the hardware cost saving entirely.

---

### B2 — Arithmetic Intensity and Roofline Analysis

**Given parameters:**

| Parameter | Value |
|---|---|
| FLOPs per inference | 45,000,000 (45 MFLOPs) |
| Memory accessed per inference | 18 MB |
| CPU peak throughput (RPi 5 Cortex-A76 NEON) | 16 GFLOP/s |
| Memory bandwidth (LPDDR4X) | 12 GB/s |

**Arithmetic Intensity:**

```
AI = FLOPs / Bytes
   = 45,000,000 / (18 x 1,024 x 1,024)
   = 45,000,000 / 18,874,368
   = 2.384 FLOP/byte
```

**Ridge point:**

```
Ridge point = Peak GFLOP/s / Peak GB/s
            = 16 / 12
            = 1.333 FLOP/byte
```

**Classification:** AI (2.384) > Ridge point (1.333) on x86 benchmark hardware.
On the actual target RPi 5 (effective bandwidth ~6.4 GB/s):

```
Ridge point on RPi 5 = 16 / 6.4 = 2.5 FLOP/byte
AI (2.384) < Ridge point (2.5) -> memory-bandwidth bound on RPi 5
```

**Optimisation implication:** A memory-bandwidth-bound model benefits most from
reducing bytes moved per inference. INT8 quantisation reduces bytes per weight from
4 to 1 (4x reduction), directly attacking the bottleneck. This is why M2 and M3
were built — not just for size, but to reduce memory pressure on the target hardware.

| Model | Size (KB) | Bytes/weight | Accuracy |
|---|---|---|---|
| M1 FP32 | 5.19 | 4 bytes | 100.00% |
| M2 PTQ INT8 | 3.38 | 1 byte (4x reduction) | 95.77% |
| M3 Pruned INT8 | 3.38 | 1 byte + 35% zeros | 94.37% |

---

*End of Phase 1 Report*
