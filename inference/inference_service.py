"""
LogiEdge Inference Service
-----------------------------
Subscribes to a truck's raw sensor topics, runs the preprocessing +
sliding-window feature pipeline, performs on-device TFLite inference,
and publishes the classification result back to MQTT. Designed to run
inside the Docker container defined in inference/Dockerfile, fully
offline apart from the local MQTT broker.

Environment variables:
    MODEL_PATH   path to the .tflite model to load (default /app/model.tflite)
    TRUCK_ID     truck identifier (default TRK-001)
    BROKER_HOST  MQTT broker host (default mosquitto, use "localhost" for local runs)
    BROKER_PORT  MQTT broker port (default 1883)

Publishes to: logibridge/trucks/{truck_id}/inference
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf

# ---------------------------------------------------------
# Use TensorFlow Lite interpreter.
# If tflite_runtime exists use it.
# Otherwise automatically use TensorFlow Lite.
# ---------------------------------------------------------
try:
    import tflite_runtime.interpreter as tflite

    Interpreter = tflite.Interpreter
    TFLITE_BACKEND = "tflite_runtime"

except ImportError:
    Interpreter = tf.lite.Interpreter
    TFLITE_BACKEND = "tensorflow.lite"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from preprocessing import SlidingWindowBuffer, load_stats, normalise  # noqa: E402

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model.tflite")
)
TRUCK_ID = os.environ.get("TRUCK_ID", "TRK-001")
BROKER_HOST = os.environ.get("BROKER_HOST", "mosquitto")
BROKER_PORT = int(os.environ.get("BROKER_PORT", "1883"))
STATS_PATH = os.environ.get(
    "STATS_PATH",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data_pipeline",
        "training_stats.npy"
    )
)

CLASS_LABELS = {0: "Normal", 1: "Warning", 2: "Critical"}


def load_interpreter(model_path: str):
    """
    Load the TensorFlow Lite model.

    On Raspberry Pi:
        Uses tflite_runtime if installed.

    On Windows / VS Code:
        Automatically falls back to TensorFlow Lite.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found:\n{model_path}\n\n"
            "Copy one of the generated models into\n"
            "inference/model.tflite"
        )

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference(interpreter, feature_vector: np.ndarray):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_data = feature_vector.reshape(1, -1)

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

    predicted_class = int(np.argmax(output))
    confidence = float(np.max(output))
    return predicted_class, confidence


class InferenceNode:
    def __init__(self):
        self.interpreter = load_interpreter(MODEL_PATH)
        self.stats = load_stats(STATS_PATH)
        self.buffer = SlidingWindowBuffer()
        self.start_time = time.time()

        self.client = mqtt.Client(client_id=f"inference-{TRUCK_ID}",
                                   callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

        self.result_topic = f"logibridge/trucks/{TRUCK_ID}/inference"

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(f"logibridge/trucks/{TRUCK_ID}/temperature")
        client.subscribe(f"logibridge/trucks/{TRUCK_ID}/vibration")
        print(f"[inference] connected to {BROKER_HOST}:{BROKER_PORT}, "
              f"model={MODEL_PATH}, backend={TFLITE_BACKEND}")

    def _on_message(self, client, userdata, msg):
        t_offset = time.time() - self.start_time
        try:
            payload = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            return

        if msg.topic.endswith("/temperature"):
            self.buffer.add_temperature(t_offset, payload["temperature_c"])
        elif msg.topic.endswith("/vibration"):
            self.buffer.add_vibration(t_offset, payload["vibration_rms_g"])

        if self.buffer.ready(t_offset):
            features = self.buffer.emit(t_offset)
            norm_features = normalise(features, self.stats).astype(np.float32)
            predicted_class, confidence = run_inference(self.interpreter, norm_features)
            self._publish_result(predicted_class, confidence)

    def _publish_result(self, predicted_class: int, confidence: float):
        result = {
            "truck_id": TRUCK_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "class_id": predicted_class,
            "class_label": CLASS_LABELS[predicted_class],
            "confidence": round(confidence, 4),
        }
        self.client.publish(self.result_topic, json.dumps(result), qos=1)
        tag = "[ALERT]" if predicted_class > 0 else "[OK]"
        print(f"{tag} {result}")

    def run_forever(self):
        self.client.loop_forever()


if __name__ == "__main__":
    node = InferenceNode()
    node.run_forever()