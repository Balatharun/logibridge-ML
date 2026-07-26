"""
LogiEdge Model Conversion (M2 - Post-Training Quantisation, Full INT8)
------------------------------------------------------------------------
Loads the trained M1 Keras model and produces a Full INT8 quantised
TFLite model using a representative dataset of >= 200 calibration
samples drawn from the normalised training feature set.

Output: training/models/m2_ptq_int8.tflite

Run:
    python training/convert_ptq.py
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from preprocessing import load_stats, normalise, FEATURE_NAMES  # noqa: E402

DATASET_CSV = os.path.join(os.path.dirname(__file__), "dataset.csv")
STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "training_stats.npy")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "m1_fp32.keras")
CALIBRATION_SAMPLES = 200


def representative_dataset_gen(X_norm: np.ndarray):
    def _gen():
        n = min(CALIBRATION_SAMPLES, len(X_norm))
        idx = np.random.choice(len(X_norm), size=n, replace=False)
        for i in idx:
            yield [X_norm[i:i + 1].astype(np.float32)]
    return _gen


def main():
    df = pd.read_csv(DATASET_CSV)
    X = df[FEATURE_NAMES].values.astype(np.float64)
    stats = load_stats(STATS_PATH)
    X_norm = np.array([normalise(row, stats) for row in X], dtype=np.float32)

    assert len(X_norm) >= CALIBRATION_SAMPLES, (
        f"Need at least {CALIBRATION_SAMPLES} samples for calibration, "
        f"found {len(X_norm)}. Regenerate the dataset with longer durations."
    )

    model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(X_norm)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8_model = converter.convert()

    out_path = os.path.join(MODELS_DIR, "m2_ptq_int8.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_int8_model)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved Full INT8 PTQ model: {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
