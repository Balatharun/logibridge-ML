"""
LogiEdge Model Benchmarking
------------------------------
Benchmarks the three model variants (M1 FP32, M2 PTQ INT8,
M3 Pruned + INT8) on five metrics:

    1. Mean inference latency (ms)      - 200 timed runs after 10 warm-up runs
    2. p95 inference latency (ms)
    3. Model file size (KB)
    4. Classification accuracy (%) on the held-out validation split
    5. Energy per inference (mJ)        - E = P x t, P estimated from psutil
                                            CPU% x laptop TDP

Outputs:
    optimisation/results/benchmark_results.csv
    optimisation/results/pareto_chart.png   (latency vs accuracy, size-coded)

Run:
    python optimisation/benchmark.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from preprocessing import load_stats, normalise, FEATURE_NAMES  # noqa: E402

DATASET_CSV = os.path.join(os.path.dirname(__file__), "..", "training", "dataset.csv")
STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "training_stats.npy")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "training", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

WARMUP_RUNS = 10
TIMED_RUNS = 200
LAPTOP_TDP_WATTS = 15.0  # adjust to match the benchmarking machine's documented TDP
SEED = 42

VARIANTS = {
    "M1_FP32": os.path.join(MODELS_DIR, "m1_fp32.tflite"),
    "M2_PTQ_INT8": os.path.join(MODELS_DIR, "m2_ptq_int8.tflite"),
    "M3_PRUNED_INT8": os.path.join(MODELS_DIR, "m3_pruned_int8.tflite"),
}


def load_validation_split():
    df = pd.read_csv(DATASET_CSV)
    X = df[FEATURE_NAMES].values.astype(np.float64)
    y = df["label"].values.astype(np.int64)
    stats = load_stats(STATS_PATH)
    X_norm = np.array([normalise(row, stats) for row in X], dtype=np.float32)
    _, X_val, _, y_val = train_test_split(X_norm, y, test_size=0.20,
                                            random_state=SEED, stratify=y)
    return X_val, y_val


def predict_one(interpreter, input_details, output_details, x_row: np.ndarray):
    input_data = x_row.reshape(1, -1)
    if input_details["dtype"] == np.int8:
        scale, zero_point = input_details["quantization"]
        input_data = (input_data / scale + zero_point).astype(np.int8)
    else:
        input_data = input_data.astype(np.float32)

    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    if output_details["dtype"] == np.int8:
        scale, zero_point = output_details["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale
    return int(np.argmax(output))


def benchmark_variant(name: str, model_path: str, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    n = len(X_val)

    # Warm-up
    for i in range(WARMUP_RUNS):
        predict_one(interpreter, input_details, output_details, X_val[i % n])

    # Timed runs
    latencies_ms = []
    cpu_start = psutil.cpu_percent(interval=None)
    t0 = time.perf_counter()
    for i in range(TIMED_RUNS):
        row = X_val[i % n]
        start = time.perf_counter()
        predict_one(interpreter, input_details, output_details, row)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    total_elapsed_s = time.perf_counter() - t0
    cpu_percent = psutil.cpu_percent(interval=None)
    avg_cpu_fraction = max(cpu_percent, cpu_start, 1.0) / 100.0

    mean_latency = float(np.mean(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))

    # Accuracy and per-class recall on full validation split
    preds = [predict_one(interpreter, input_details, output_details, x_row) for x_row in X_val]
    correct = sum(int(p == y) for p, y in zip(preds, y_val))
    accuracy_pct = 100.0 * correct / len(y_val)
    critical_recall_pct = 100.0 * recall_score(y_val, preds, labels=[2], average="micro")

    size_kb = os.path.getsize(model_path) / 1024

    # Energy per inference: E = P x t
    power_watts = avg_cpu_fraction * LAPTOP_TDP_WATTS
    mean_latency_s = mean_latency / 1000.0
    energy_mj = power_watts * mean_latency_s * 1000.0  # W * s * 1000 = mJ

    return {
        "variant": name,
        "mean_latency_ms": round(mean_latency, 4),
        "p95_latency_ms": round(p95_latency, 4),
        "model_size_kb": round(size_kb, 2),
        "accuracy_pct": round(accuracy_pct, 2),
        "critical_recall_pct": round(critical_recall_pct, 2),
        "energy_per_inference_mj": round(energy_mj, 4),
    }


def plot_pareto(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = df["model_size_kb"] / df["model_size_kb"].max() * 800 + 100
    scatter = ax.scatter(df["mean_latency_ms"], df["accuracy_pct"],
                          s=sizes, alpha=0.7, edgecolors="black")
    for _, row in df.iterrows():
        ax.annotate(
            f"{row['variant']}\n{row['model_size_kb']:.0f} KB",
            (row["mean_latency_ms"], row["accuracy_pct"]),
            textcoords="offset points", xytext=(8, 8), fontsize=9,
        )
    ax.set_xlabel("Mean inference latency (ms)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("LogiEdge Model Variants: Latency vs Accuracy (bubble size = model size)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved Pareto chart: {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_val, y_val = load_validation_split()

    results = []
    for name, path in VARIANTS.items():
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping {name}. "
                  f"Run the corresponding training/conversion script first.")
            continue
        print(f"Benchmarking {name} ...")
        result = benchmark_variant(name, path, X_val, y_val)
        results.append(result)
        print(result)

    if not results:
        print("No models found to benchmark.")
        return

    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark results: {csv_path}")
    print(df.to_string(index=False))

    plot_pareto(df, os.path.join(RESULTS_DIR, "pareto_chart.png"))


if __name__ == "__main__":
    main()
