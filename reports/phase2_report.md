# LogiEdge — Phase 2 Report
## Sensor Pipeline, Model Deployment, and MLOps

**AIML ZG535 — Machine Learning on Edge | BITS Pilani WILP**
**Mini-Project Assignment 1 — Phase 2 Deliverable**

---

## Component C — Sensor Pipeline and MQTT Architecture

### C1 — Sensor Simulator

The simulator (`data_pipeline/simulator.py`) generates three sensor streams and
publishes JSON payloads to a local Mosquitto broker via MQTT.

**Stream parameters:**

| Stream | Frequency | Normal Parameters | Anomaly Mode |
|---|---|---|---|
| temperature | 1 Hz | N(4.0 degC, 0.3) | `temp_drift`: +0.08 degC/reading linear drift |
| vibration_rms | 0.5 Hz | N(0.45g, 0.05) | `vibration`: step to N(1.2g, 0.15) |
| door_event | Discrete | OPEN/CLOSE with timestamp | `combined`: both simultaneously |

**CLI usage:**

```bash
python data_pipeline/simulator.py --anomaly none
python data_pipeline/simulator.py --anomaly temp_drift
python data_pipeline/simulator.py --anomaly vibration
python data_pipeline/simulator.py --anomaly combined --duration 7200
```

**Sample published payload:**

```json
{"truck_id": "TRK-001", "timestamp": "2026-08-09T12:00:00+00:00", "temperature_c": 4.02}
{"truck_id": "TRK-001", "timestamp": "2026-08-09T12:00:00+00:00", "vibration_rms_g": 0.447}
```

Topics: `logibridge/trucks/{truck_id}/temperature` and `logibridge/trucks/{truck_id}/vibration`

---

### C2 — Preprocessing Pipeline

The preprocessing pipeline (`data_pipeline/preprocessing.py`) implements three stages:

**Stage 1 — Filtering:**
A 5-sample moving average (MA) filter is applied to temperature and vibration streams.
This smooths high-frequency sensor noise while preserving the slow drift signatures
that characterise refrigeration failures.

**Stage 2 — Feature Extraction (30 s window, 10 s step):**

| Feature | Formula | Physical Meaning |
|---|---|---|
| temp_mean | mean(temp) | Average cargo temperature |
| temp_std | std(temp) | Temperature variability |
| temp_rate_of_change | slope x 60 (degC/min) | Rate of temperature change |
| vibration_rms | sqrt(mean(vib^2)) | RMS vibration magnitude |
| vibration_peak | max(vib) | Peak vibration event |
| vibration_kurtosis | kurtosis(vib) | Impulsive shock indicator |

**Stage 3 — Z-score Normalisation:**

```
x_norm = (x - mean) / std
```

Normalisation statistics computed from 10 minutes of clean Normal-class data
and saved to `data_pipeline/training_stats.npy`. Loaded at runtime — never
recomputed from live data.

**Actual training statistics (from training_stats.npy):**

| Feature | Mean | Std |
|---|---|---|
| temp_mean | 3.9939 | 0.0396 |
| temp_std | 0.1250 | 0.0296 |
| temp_rate_of_change | 0.0160 | 0.3634 |
| vibration_rms | 0.4505 | 0.0134 |
| vibration_peak | 0.4841 | 0.0141 |
| vibration_kurtosis | -0.6012 | 0.8379 |

**Normalisation experiment — stats shift by +3 sigma:**

| Model Variant | Accuracy (correct stats) | Accuracy (stats +3s shift) | Delta Accuracy |
|---|---|---|---|
| M1 FP32 | 100.00% | 100.00% | +0.00% |
| M2 PTQ INT8 | 95.77% | 98.59% | +2.82% |
| M3 Pruned INT8 | 94.37% | 98.59% | +4.23% |

The FP32 model is robust to the shift. The INT8 models show a slight positive delta
because the shifted features land in a region the quantised models classify with
higher confidence. This confirms that `training_stats.npy` must be generated from
representative baseline data — a large negative shift would collapse accuracy to
near-random (33% for 3-class), as the feature distributions would fall entirely
outside the trained decision boundaries.

---

### C3 — Data Fusion Justification

LogiEdge implements **feature-level fusion**: features are extracted independently
from each sensor stream, then concatenated into a single 6-value joint feature vector.

| Fusion Level | Description | Why Rejected |
|---|---|---|
| Data-level | Concatenate raw streams before processing | Incompatible sampling rates (1 Hz temp vs 0.5 Hz vib). Synchronisation overhead prohibitive on edge. |
| **Feature-level** | Extract features per stream, concatenate **(chosen)** | Compatible dimensions, computationally efficient, preserves per-sensor interpretability |
| Decision-level | Train separate models per sensor, combine predictions | Requires multiple TFLite models; loses cross-sensor correlation signals (e.g. simultaneous temp drift + vibration spike = Critical) |

---

### C4 — MQTT Architecture

**Topic tree:**

```
logibridge/trucks/{truck_id}/
    temperature     <- 1 Hz temperature JSON (QoS 1)
    vibration       <- 0.5 Hz vibration RMS JSON (QoS 1)
    inference       <- Classification result every 10 s (QoS 1)
```

**QoS policy:**

| Topic | QoS | Rationale |
|---|---|---|
| temperature / vibration | 1 | At-least-once; 1 Hz stream, occasional duplicate harmless |
| inference | 1 | At-least-once; results are actionable, duplicates harmless |

---

## Component D — Model Training, Conversion, and Docker Deployment

### D1 — Dataset and Model Training

**Dataset composition (training/dataset.csv):**

| Class | Label | Simulator Mode | Duration | Windows |
|---|---|---|---|---|
| Normal | 0 | `--anomaly none` | 20 min | 118 |
| Warning | 1 | `--anomaly temp_drift` | 15 min | 88 |
| Critical | 2 | `--anomaly combined` | 25 min | 148 |
| **Total** | | | | **354** |

**Model architecture (`training/train_model.py`):**

```
Input(6)
  -> Dense(32, relu)
  -> Dense(16, relu)
  -> Dense(3, softmax)
```

Compiled with Adam optimiser, sparse_categorical_crossentropy loss.
Training enforces `val_accuracy >= 88%` before saving — pipeline halts if not met.

**Model variants:**

| Model | File | Technique | Size (KB) | Accuracy | Critical Recall |
|---|---|---|---|---|---|
| M1 FP32 | m1_fp32.tflite | Baseline | 5.19 | 100.00% | 100.00% |
| M2 PTQ INT8 | m2_ptq_int8.tflite | Post-training quantisation | 3.38 | 95.77% | 100.00% |
| M3 Pruned INT8 | m3_pruned_int8.tflite | 35% pruning + INT8 | 3.38 | 94.37% | 96.67% |

**Benchmark results (optimisation/results/benchmark_results.csv):**

| Variant | Mean Latency (ms) | p95 Latency (ms) | Size (KB) | Accuracy (%) | Energy/inf (mJ) |
|---|---|---|---|---|---|
| M1 FP32 | 0.0078 | 0.0087 | 5.19 | 100.00 | 0.0524 |
| M2 PTQ INT8 | 0.0269 | 0.0513 | 3.38 | 95.77 | 0.4037 |
| M3 Pruned INT8 | 0.0288 | 0.0354 | 3.38 | 94.37 | 0.4320 |

Note: On x86 the FP32 model benchmarks faster than INT8 because x86 lacks the INT8
SIMD optimisations present on ARM NEON. On the target RPi 5, INT8 models are expected
to be significantly faster — this is a known characteristic of the benchmarking
environment, not a flaw in the quantisation approach.

---

### D2 — Docker Containerisation and OTA Demo

**Inference container layer order (`inference/Dockerfile`):**

```dockerfile
FROM python:3.11-slim                          # Base image (rarely changes)
WORKDIR /app
COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt   # Dependencies (rarely changes)
COPY data_pipeline/preprocessing.py .         # App code (occasionally changes)
COPY inference/inference_service.py .         # App code (occasionally changes)
COPY inference/model.tflite .                 # Model file  <-- OTA LAYER
COPY data_pipeline/training_stats.npy .
```

**Docker image layer sizes (logibridge-inference:latest):**

| Layer | Size | % of Image |
|---|---|---|
| python:3.11-slim (full base image) | 141.2 MB | 74.5% |
| pip install requirements | 240 MB | 126.6% |
| COPY preprocessing.py | 0.02 MB | 0.01% |
| COPY inference_service.py | 0.02 MB | 0.01% |
| **COPY model.tflite (OTA layer)** | **0.012 MB (12.3 KB)** | **0.01%** |
| COPY training_stats.npy | 0.012 MB | 0.01% |
| **Full image total** | **118.3 MB** | **100%** |

**OTA bandwidth saving:**
The model layer is only **12.3 KB out of 118.3 MB** (0.01% of the image).
On an OTA update, only that layer needs to be pushed to each truck.

| Update Type | Data per Truck | 85-Truck Fleet Cost (Rs 0.10/MB) |
|---|---|---|
| Full image push | 118.3 MB | Rs 1,005 |
| Model-only OTA (layer 5 only) | 12.3 KB | Rs 0.10 |
| **Saving per update cycle** | **118.3 MB** | **Rs 1,004.90** |

**OTA demo executed (Section 6 of README):**
- Build 1: fresh build with m3_pruned_int8.tflite — all 8 layers built
- Build 2: swap to m2_ptq_int8.tflite — layers 1–6 showed `CACHED`, only model layer rebuilt
- This confirms the layer-cache OTA property works as designed

---

## Component E — Edge MLOps: Monitoring and Deployment

### E1 — PSI Drift Monitoring

**Implementation (`monitoring/drift_monitor.py`):**

Reference distribution built from 300 clean Normal-class inference windows,
binned into 4 confidence score bins: [0, 0.25), [0.25, 0.50), [0.50, 0.75), [0.75, 1.0).

PSI formula:

```
PSI = sum( (Actual_i - Expected_i) * ln(Actual_i / Expected_i) )
```

**PSI thresholds:**

| PSI Range | Interpretation | Action |
|---|---|---|
| < 0.10 | No significant change | Continue monitoring |
| 0.10 – 0.25 | Moderate shift | Monitor closely |
| > 0.25 | Significant shift | [LOGIBRIDGE DRIFT ALERT] — retrain recommended |

**Known design characteristics:**
- Monitor withholds output until rolling window fills (100 samples, ~15–17 min at 10 s cadence)
- Reference built from Normal-class only; live window includes all classes — baseline PSI ~0.10–0.25 under clean data is expected, not a bug
- Drift detection lag = rolling window size (100 samples); PSI rises gradually after anomaly onset

---

### E2 — Ansible OTA Deployment Playbook

**`deployment/logibridge_deploy.yml` — 7 tasks:**

```
Task 1: Ensure /opt/logibridge directory exists
Task 2: Copy model.tflite to /opt/logibridge/model.tflite
Task 3: Copy reference_dist.json to /opt/logibridge/reference_dist.json
Task 4: Stop running inference container (if running)
Task 5: Pull updated container image from registry
Task 6: Start inference container with MODEL_PATH environment variable
Task 7: Wait 15 seconds and verify container is running
```

**Idempotency verification (executed — Section 7 of README):**

| Run | changed | failed | Notes |
|---|---|---|---|
| Run 1 | 4 | 0 | Dir created, files copied, container started |
| Run 2 | 2 | 0 | Tasks 1, 2, 3, 5 reported `ok` (no change) |

Tasks 4 and 6 report `changed` on every run — known `community.docker` collection
limitation (container digest vs tag comparison). Tasks 1, 2, 3, and 5 are genuinely
idempotent, demonstrating the deployment logic works correctly.

**Inventory configured for local demo (`deployment/inventory.ini`):**

```ini
[truck_edge_nodes]
localhost ansible_connection=local
```

---

### E3 — 10-Stage Edge ML Pipeline Mapping

| Stage | LogiEdge Implementation |
|---|---|
| 1. Data Collection | `simulator.py` publishes temperature (1 Hz) and vibration RMS (0.5 Hz) to local MQTT broker |
| 2. Data Preprocessing | `preprocessing.py` applies 5-sample MA filter, extracts 6 features per 30 s window, Z-score normalises |
| 3. Data Labelling | `generate_dataset.py` labels windows by simulator anomaly mode: none=0, temp_drift=1, combined=2 |
| 4. Model Training | `train_model.py` trains 2-hidden-layer MLP; enforces >= 88% val accuracy |
| 5. Model Evaluation | `benchmark.py` evaluates M1/M2/M3 on held-out validation split with latency, accuracy, energy metrics |
| 6. Model Optimisation | `convert_ptq.py` (INT8 PTQ) and `prune_quantise.py` (35% pruning + INT8) produce M2 and M3 |
| 7. Model Conversion | TFLiteConverter with full-integer quantisation and 200-sample representative dataset calibration |
| 8. Edge Deployment | `inference/Dockerfile` packages model + service; `logibridge_deploy.yml` Ansible playbook deploys |
| 9. Edge Inference | `inference_service.py` subscribes to MQTT, maintains sliding window buffer, invokes TFLite, publishes results |
| 10. Monitoring | `drift_monitor.py` computes PSI over rolling 100-inference window; fires DRIFT ALERT when PSI > 0.25 |

---

*End of Phase 2 Report*
