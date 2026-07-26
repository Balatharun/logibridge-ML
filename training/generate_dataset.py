"""
LogiEdge Dataset Generator
----------------------------
Generates a labelled training dataset by replaying the same sensor
statistics used in data_pipeline/simulator.py, without requiring a live
MQTT broker. This keeps dataset generation fast and fully reproducible.

Class 0 (Normal)   -> --anomaly none        -> 20 minutes -> ~120 windows
Class 1 (Warning)  -> --anomaly temp_drift  -> 15 minutes -> ~90 windows
Class 2 (Critical) -> --anomaly combined    -> 15 minutes -> ~90 windows

Output: training/dataset.csv  (columns: 6 features + label)
Also saves data_pipeline/training_stats.npy computed from 10 minutes of
clean Normal-class output only, as required by the preprocessing spec.

Run:
    python training/generate_dataset.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from preprocessing import (  # noqa: E402
    moving_average, extract_features, compute_and_save_stats, FEATURE_NAMES,
)

SETPOINT_C = 4.0
TEMP_STD_NORMAL = 0.3
TEMP_DRIFT_PER_READING = 0.08

VIB_MEAN_NORMAL = 0.45
VIB_STD_NORMAL = 0.05
VIB_MEAN_ANOMALY = 1.2
VIB_STD_ANOMALY = 0.15

WINDOW_SECONDS = 30
STEP_SECONDS = 10
TEMP_HZ = 1.0
VIB_HZ = 0.5

SEED = 42
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "dataset.csv")
STATS_OUT = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "training_stats.npy")


def simulate_raw_streams(duration_s: int, anomaly_mode: str, rng: np.random.Generator):
    n_temp = int(duration_s * TEMP_HZ)
    n_vib = int(duration_s * VIB_HZ)

    temp_times = np.arange(n_temp) / TEMP_HZ
    vib_times = np.arange(n_vib) / VIB_HZ

    if anomaly_mode in ("temp_drift", "combined"):
        drift = TEMP_DRIFT_PER_READING * np.arange(n_temp)
    else:
        drift = np.zeros(n_temp)
    temp_series = rng.normal(SETPOINT_C, TEMP_STD_NORMAL, n_temp) + drift

    if anomaly_mode in ("vibration", "combined"):
        vib_series = rng.normal(VIB_MEAN_ANOMALY, VIB_STD_ANOMALY, n_vib)
    else:
        vib_series = rng.normal(VIB_MEAN_NORMAL, VIB_STD_NORMAL, n_vib)
    vib_series = np.clip(vib_series, 0, None)

    return temp_times, temp_series, vib_times, vib_series


def windows_from_streams(temp_times, temp_series, vib_times, vib_series, duration_s):
    """Slide a 30s/10s window over the generated streams and extract features."""
    temp_series = moving_average(temp_series)
    vib_series = moving_average(vib_series)

    rows = []
    t = WINDOW_SECONDS
    while t <= duration_s:
        t_mask = (temp_times > t - WINDOW_SECONDS) & (temp_times <= t)
        v_mask = (vib_times > t - WINDOW_SECONDS) & (vib_times <= t)
        temp_w = temp_series[t_mask]
        temp_t = temp_times[t_mask]
        vib_w = vib_series[v_mask]
        if len(temp_w) > 1:
            feats = extract_features(temp_w, temp_t, vib_w)
            rows.append(feats)
        t += STEP_SECONDS
    return np.array(rows)


def main():
    rng = np.random.default_rng(SEED)

    class_specs = [
    (0, "none", 20 * 60),
    (1, "temp_drift", 15 * 60),
    (2, "combined", 25 * 60),
]
    

    all_rows = []
    normal_feature_rows_for_stats = []

    for label, anomaly_mode, duration_s in class_specs:
        temp_t, temp_s, vib_t, vib_s = simulate_raw_streams(duration_s, anomaly_mode, rng)
        feats = windows_from_streams(temp_t, temp_s, vib_t, vib_s, duration_s)
        for row in feats:
            all_rows.append(list(row) + [label])
        print(f"class={label} mode={anomaly_mode} windows={len(feats)}")

        if label == 0:
            # First 10 minutes of clean Normal output -> normalisation reference stats
            ten_min_mask_end = 10 * 60
            feats_10min = windows_from_streams(temp_t, temp_s, vib_t, vib_s, ten_min_mask_end)
            normal_feature_rows_for_stats = feats_10min

    df = pd.DataFrame(all_rows, columns=FEATURE_NAMES + ["label"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved dataset: {OUTPUT_CSV} ({len(df)} rows)")

    stats = compute_and_save_stats(np.array(normal_feature_rows_for_stats), path=STATS_OUT)
    print(f"Saved normalisation stats: {STATS_OUT}")
    print("mean:", stats["mean"])
    print("std :", stats["std"])


if __name__ == "__main__":
    main()
