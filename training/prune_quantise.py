"""
LogiEdge Model Optimisation (M3 - 35% Structured Pruning + Full INT8 PTQ)
----------------------------------------------------------------------------
Applies magnitude-based structured pruning to the trained MLP using a
PolynomialDecay sparsity schedule (target sparsity 0.35), fine-tunes the
pruned model, strips pruning wrappers, then applies the same Full INT8
PTQ procedure used for M2.

Output: training/models/m3_pruned_int8.tflite

Run:
    python training/prune_quantise.py
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from preprocessing import load_stats, normalise, FEATURE_NAMES  # noqa: E402

DATASET_CSV = os.path.join(os.path.dirname(__file__), "dataset.csv")
STATS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data_pipeline", "training_stats.npy"
)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "m1_fp32.keras")

TARGET_SPARSITY = 0.35
CALIBRATION_SAMPLES = 200
SEED = 42


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(DATASET_CSV)

    X = df[FEATURE_NAMES].values.astype(np.float64)
    y = df["label"].values.astype(np.int64)

    stats = load_stats(STATS_PATH)
    X_norm = np.array(
        [normalise(row, stats) for row in X],
        dtype=np.float32,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_norm,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    # ------------------------------------------------------------------
    # Load trained FP32 model
    # ------------------------------------------------------------------
    base_model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    batch_size = 16
    epochs = 10

    end_step = int(np.ceil(len(X_train) / batch_size)) * epochs

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=TARGET_SPARSITY,
            begin_step=0,
            end_step=end_step,
        )
    }

    # ------------------------------------------------------------------
    # Apply pruning
    # ------------------------------------------------------------------
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        base_model,
        **pruning_params,
    )

    pruned_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep()
    ]

    pruned_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # ------------------------------------------------------------------
    # Strip pruning wrappers
    # ------------------------------------------------------------------
    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    # IMPORTANT:
    # strip_pruning() returns an UNCOMPILED Keras model.
    # Recompile before evaluating or converting.
    final_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    val_loss, val_acc = final_model.evaluate(
        X_val,
        y_val,
        verbose=0,
    )

    print(f"Pruned model validation accuracy: {val_acc:.4f}")

    # ------------------------------------------------------------------
    # Representative dataset
    # ------------------------------------------------------------------
    def representative_dataset_gen():
        n = min(CALIBRATION_SAMPLES, len(X_norm))
        idx = np.random.choice(len(X_norm), size=n, replace=False)

        for i in idx:
            yield [X_norm[i:i + 1].astype(np.float32)]

    # ------------------------------------------------------------------
    # Convert to Full INT8
    # ------------------------------------------------------------------
    converter = tf.lite.TFLiteConverter.from_keras_model(final_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(MODELS_DIR, exist_ok=True)

    out_path = os.path.join(
        MODELS_DIR,
        "m3_pruned_int8.tflite",
    )

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(out_path) / 1024

    print(f"Saved pruned + INT8 model: {out_path}")
    print(f"Model size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()