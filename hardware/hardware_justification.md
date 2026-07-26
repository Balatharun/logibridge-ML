# Task B1/B2 — Hardware Justification (template)

## Constraint Triangle (Power / Compute / Cost) — fill in per option

| Option | Power (10W budget) | Compute headroom for 90s SLA | Fleet cost (85 / 265 trucks) |
|---|---|---|---|
| 1. Raspberry Pi 5 + AI HAT+ (13 TOPS) | 7.5 W — fits | Ample for a 6-value MLP | ₹12.75L / ₹39.75L |
| 2. Jetson Orin Nano Super (67 TOPS) | 15 W typical — exceeds 10W budget under load | Overkill for this workload | ₹38.25L / ₹119.25L |
| 3. STM32H7 MCU | 0.4 W — large margin | Sufficient for a small MLP, tight for future CV models | ₹2.975L / ₹9.275L |

State the dominant constraint vertex (this workload's model is tiny — 45 MFLOPs — so cost and
power dominate over raw compute) and argue for **Option 1** as the balanced choice unless your
own analysis leads elsewhere; justify against Options 2 and 3 explicitly.

## Arithmetic Intensity / Roofline (Task B2)

Given: 45 MFLOPs/inference, 18 MB accessed/inference, CPU peak 16 GFLOP/s, bandwidth 12 GB/s.

```
Arithmetic Intensity (AI) = FLOPs / Bytes = 45,000,000 / (18 * 1024 * 1024)
Ridge point = Peak GFLOP/s / Peak GB/s = 16 / 12
```

Compute both values, compare AI against the ridge point, classify the model as memory-bandwidth
bound or compute-bound, and state the optimisation implication (e.g. INT8 quantisation reduces
bytes moved per inference, which helps a memory-bound model far more than a compute-bound one —
tie this back to why M2/M3 in this repository were built).
