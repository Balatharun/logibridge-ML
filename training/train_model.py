"""
LogiEdge Model Training (M1 - FP32 baseline)
----------------------------------------------
Trains a 2-hidden-layer MLP (32, 16 units, ReLU) on the 6-value feature
vectors produced by generate_dataset.py, and classifies cargo state into
Normal (0) / Warning (1) / Critical (2).

Requires validation accuracy >= 88%. If the threshold is not met, the
script exits non-zero rather than silently saving an unfit model.

Outputs:
    training/models/m1_fp32.keras         (Keras model)
    training/models/m1_fp32.tflite        (TFLite float32 export)
    training/models/model.tflite          (copy used as the default runtime model)

Run:
    python training/train_model.py
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from preprocessing import load_stats, normalise, FEATURE_NAMES  # noqa: E402

DATASET_CSV = os.path.join(os.path.dirname(__file__), "dataset.csv")
STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "training_stats.npy")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
ACCURACY_THRESHOLD = 0.88
SEED = 42


def build_model(input_dim: int, num_classes: int = 3) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(DATASET_CSV)
    X = df[FEATURE_NAMES].values.astype(np.float64)
    y = df["label"].values.astype(np.int64)

    stats = load_stats(STATS_PATH)
    X_norm = np.array([normalise(row, stats) for row in X], dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y, test_size=0.20, random_state=SEED, stratify=y
    )

    model = build_model(input_dim=X_norm.shape[1])
    model.summary()

    model.fit(
      X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=16,
    class_weight={
        0: 1.0,   # Normal
        1: 1.0,   # Warning
        2: 3.0    # Critical - higher penalty for misclassification
    },
    verbose=2,
    )

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    val_accuracy = accuracy_score(y_val, y_pred)
    print(f"\nValidation accuracy: {val_accuracy:.4f}")
    print(classification_report(y_val, y_pred, target_names=["Normal", "Warning", "Critical"]))

    if val_accuracy < ACCURACY_THRESHOLD:
        print(f"FAILED: validation accuracy {val_accuracy:.4f} is below the "
              f"required {ACCURACY_THRESHOLD:.2f} threshold for pharmaceutical "
              f"cold-chain monitoring. Revisit features or architecture.")
        sys.exit(1)

    # Save Keras model
    keras_path = os.path.join(MODELS_DIR, "m1_fp32.keras")
    model.save(keras_path)
    print(f"Saved Keras model: {keras_path}")

    # Export FP32 TFLite (M1 baseline)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    m1_path = os.path.join(MODELS_DIR, "m1_fp32.tflite")
    with open(m1_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite FP32 model: {m1_path}")

    # Default runtime copy
    default_path = os.path.join(MODELS_DIR, "model.tflite")
    with open(default_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved default runtime model: {default_path}")


if __name__ == "__main__":
    main()
