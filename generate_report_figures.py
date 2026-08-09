"""
LogiBridge Report Figures
--------------------------
Generates two PNG figures for the report:

1. reports/stats_shift_table.png
   - Accuracy of each model variant under:
       a) correct normalisation stats
       b) stats shifted by +3σ (simulates sensor drift / wrong stats)
   - Accuracy delta column

2. reports/docker_size_table.png
   - Full base image size vs model layer size

Run:
    python generate_report_figures.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

ROOT       = os.path.dirname(os.path.abspath(__file__))
STATS_PATH = os.path.join(ROOT, "data_pipeline", "training_stats.npy")
DATASET    = os.path.join(ROOT, "training", "dataset.csv")
MODELS_DIR = os.path.join(ROOT, "training", "models")
REPORTS    = os.path.join(ROOT, "reports")
SEED       = 42

sys.path.insert(0, os.path.join(ROOT, "data_pipeline"))
from preprocessing import load_stats, normalise, FEATURE_NAMES  # noqa: E402

VARIANTS = {
    "M1 FP32":        os.path.join(MODELS_DIR, "m1_fp32.tflite"),
    "M2 PTQ INT8":    os.path.join(MODELS_DIR, "m2_ptq_int8.tflite"),
    "M3 Pruned INT8": os.path.join(MODELS_DIR, "m3_pruned_int8.tflite"),
}

# Docker layer sizes (bytes) from `docker history`
# Base image = python:3.11-slim = debian OS + Python build + apt packages
BASE_IMAGE_BYTES = 87_400_000 + 48_800_000 + 4_950_000 + 16_400  # debian + python + apt + symlinks
DOCKER_SIZES = {
    "python:3.11-slim (full base image)": BASE_IMAGE_BYTES,
    "pip install requirements":           240_000_000,
    "COPY requirements.txt":                  12_300,
    "WORKDIR /app":                            8_190,
    "COPY preprocessing.py":                  16_400,
    "COPY inference_service.py":              16_400,
    "COPY model.tflite  <-- OTA layer":       12_300,
    "COPY training_stats.npy":                12_300,
}
FULL_IMAGE_BYTES = 118_275_081


# ── helpers ──────────────────────────────────────────────────────────────────

def load_val_split(stats):
    df = pd.read_csv(DATASET)
    X  = df[FEATURE_NAMES].values.astype(np.float64)
    y  = df["label"].values.astype(np.int64)
    X_norm = np.array([normalise(r, stats) for r in X], dtype=np.float32)
    _, X_val, _, y_val = train_test_split(
        X_norm, y, test_size=0.20, random_state=SEED, stratify=y)
    return X_val, y_val


def run_tflite(model_path, X_val):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    preds = []
    for row in X_val:
        data = row.reshape(1, -1)
        if inp["dtype"] == np.int8:
            sc, zp = inp["quantization"]
            data = (data / sc + zp).astype(np.int8)
        interp.set_tensor(inp["index"], data)
        interp.invoke()
        o = interp.get_tensor(out["index"])[0]
        if out["dtype"] == np.int8:
            sc, zp = out["quantization"]
            o = (o.astype(np.float32) - zp) * sc
        preds.append(int(np.argmax(o)))
    return np.array(preds)


def accuracy(preds, y_val):
    return 100.0 * np.mean(preds == y_val)


def save_table_png(df, title, out_path, col_widths=None):
    fig_h = 0.45 * (len(df) + 2)
    fig, ax = plt.subplots(figsize=(10, max(fig_h, 2.2)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    # header style
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # highlight delta column if present
    delta_col = None
    for j, c in enumerate(df.columns):
        if "Δ" in c or "Delta" in c.lower():
            delta_col = j
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = tbl[i, j]
            cell.set_facecolor("#ecf0f1" if i % 2 == 0 else "white")
            if delta_col is not None and j == delta_col:
                try:
                    val = float(str(df.values[i-1][j]).replace("%",""))
                    cell.set_facecolor("#fadbd8" if val < -2 else "#d5f5e3" if abs(val) < 1 else "#fef9e7")
                except ValueError:
                    pass
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 1: stats shift accuracy table ─────────────────────────────────────

def figure_stats_shift():
    stats_correct = load_stats(STATS_PATH)

    # shifted stats: mean += 3σ  (simulates sensor drift or wrong calibration)
    stats_shifted = {
        "mean": stats_correct["mean"] + 3 * stats_correct["std"],
        "std":  stats_correct["std"],
    }

    print("Correct stats  mean:", np.round(stats_correct["mean"], 4))
    print("Shifted stats  mean:", np.round(stats_shifted["mean"], 4))

    X_correct, y_val = load_val_split(stats_correct)
    X_shifted, _     = load_val_split(stats_shifted)

    rows = []
    for name, path in VARIANTS.items():
        acc_c = accuracy(run_tflite(path, X_correct), y_val)
        acc_s = accuracy(run_tflite(path, X_shifted), y_val)
        delta = acc_s - acc_c
        rows.append([name,
                     f"{acc_c:.2f}%",
                     f"{acc_s:.2f}%",
                     f"{delta:+.2f}%"])

    df = pd.DataFrame(rows, columns=[
        "Model Variant",
        "Accuracy (correct stats)",
        "Accuracy (stats +3s shift)",
        "Delta Accuracy",
    ])
    print("\n--- Stats Shift Accuracy Table ---")
    print(df.to_string(index=False))

    save_table_png(
        df,
        "Accuracy: Correct Normalisation Stats vs. Stats Shifted by +3s",
        os.path.join(REPORTS, "stats_shift_table.png"),
        col_widths=[0.22, 0.26, 0.28, 0.18],
    )
    return df


# ── Figure 2: Docker image size table ────────────────────────────────────────

def figure_docker_sizes():
    rows = []
    for layer, size_b in DOCKER_SIZES.items():
        size_mb = size_b / 1_048_576
        pct     = 100.0 * size_b / FULL_IMAGE_BYTES
        rows.append([layer, f"{size_mb:.2f} MB", f"{pct:.1f}%"])

    # totals row
    rows.append(["== FULL IMAGE TOTAL ==",
                 f"{FULL_IMAGE_BYTES/1_048_576:.2f} MB", "100.0%"])

    df = pd.DataFrame(rows, columns=["Layer", "Size", "% of Image"])
    print("\n--- Docker Image Size Table ---")
    print(df.to_string(index=False))

    save_table_png(
        df,
        "Docker Image Layer Sizes  (logibridge-inference:latest - 118.3 MB total)",
        os.path.join(REPORTS, "docker_size_table.png"),
        col_widths=[0.52, 0.24, 0.18],
    )
    return df


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(REPORTS, exist_ok=True)
    df_shift  = figure_stats_shift()
    df_docker = figure_docker_sizes()
    print("\nAll figures saved to reports/")
