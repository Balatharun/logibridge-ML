"""
LogiEdge Drift Monitor (Population Stability Index)
------------------------------------------------------
Monitors the inference confidence-score distribution published on
logibridge/trucks/{truck_id}/inference and computes PSI against a
reference distribution captured from 300 clean Normal-class inferences.

PSI bins (confidence score): [0, 0.25), [0.25, 0.50), [0.50, 0.75), [0.75, 1.0]

PSI = sum( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )

Interpretation (standard thresholds):
    PSI < 0.10            : no significant drift
    0.10 <= PSI <= 0.25   : moderate drift, monitor
    PSI > 0.25            : significant drift -> [LOGIBRIDGE DRIFT ALERT]

Run:
    Step 1 - build the reference distribution (300 clean windows):
        python monitoring/drift_monitor.py --mode build-reference --truck-id TRK-001

    Step 2 - start live monitoring (rolling window of last 100 inferences,
             recomputed every 60 seconds):
        python monitoring/drift_monitor.py --mode monitor --truck-id TRK-001
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np
import paho.mqtt.client as mqtt

BINS = [0.0, 0.25, 0.50, 0.75, 1.0]
ROLLING_WINDOW = 100
CHECK_INTERVAL_SECONDS = 60
PSI_ALERT_THRESHOLD = 0.25
REFERENCE_SAMPLE_SIZE = 300

MONITORING_DIR = os.path.dirname(__file__)
REFERENCE_PATH = os.path.join(MONITORING_DIR, "reference_dist.json")

EPS = 1e-6


def bin_distribution(scores: list) -> np.ndarray:
    counts, _ = np.histogram(scores, bins=BINS)
    total = max(len(scores), 1)
    return counts / total


def compute_psi(expected_pct: np.ndarray, actual_pct: np.ndarray) -> float:
    expected = np.clip(expected_pct, EPS, None)
    actual = np.clip(actual_pct, EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def build_reference(truck_id: str, broker_host: str, broker_port: int):
    scores = []

    def on_connect(client, userdata, flags, reason_code, properties=None):
        client.subscribe(f"logibridge/trucks/{truck_id}/inference")
        print(f"[drift_monitor] collecting {REFERENCE_SAMPLE_SIZE} clean Normal-class "
              f"inferences to build reference_dist.json ...")

    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        if payload.get("class_id") == 0:
            scores.append(payload["confidence"])

    client = mqtt.Client(client_id="drift-ref-builder",
                          callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker_host, broker_port, keepalive=60)
    client.loop_start()

    while len(scores) < REFERENCE_SAMPLE_SIZE:
        time.sleep(1)
        print(f"  collected {len(scores)}/{REFERENCE_SAMPLE_SIZE}", end="\r")

    client.loop_stop()
    client.disconnect()

    reference_pct = bin_distribution(scores[:REFERENCE_SAMPLE_SIZE])
    with open(REFERENCE_PATH, "w") as f:
        json.dump({"bins": BINS, "distribution": reference_pct.tolist(),
                    "n_samples": REFERENCE_SAMPLE_SIZE}, f, indent=2)
    print(f"\n[drift_monitor] saved reference distribution -> {REFERENCE_PATH}")


def monitor(truck_id: str, broker_host: str, broker_port: int):
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)
    expected_pct = np.array(reference["distribution"])

    rolling_scores = deque(maxlen=ROLLING_WINDOW)
    last_check = time.time()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        client.subscribe(f"logibridge/trucks/{truck_id}/inference")
        print(f"[drift_monitor] monitoring live PSI for {truck_id} "
              f"(rolling window={ROLLING_WINDOW}, check every {CHECK_INTERVAL_SECONDS}s)")

    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        rolling_scores.append(payload["confidence"])

    client = mqtt.Client(client_id="drift-monitor",
                          callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker_host, broker_port, keepalive=60)
    client.loop_start()

    try:
        while True:
            time.sleep(1)
            if (time.time() - last_check >= CHECK_INTERVAL_SECONDS
                    and len(rolling_scores) >= ROLLING_WINDOW):
                actual_pct = bin_distribution(list(rolling_scores))
                psi = compute_psi(expected_pct, actual_pct)
                print(f"[drift_monitor] n={len(rolling_scores)} PSI={psi:.3f}")
                if psi > PSI_ALERT_THRESHOLD:
                    print(f"[LOGIBRIDGE DRIFT ALERT] PSI={psi:.3f}")
                last_check = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="LogiEdge PSI drift monitor")
    parser.add_argument("--mode", choices=["build-reference", "monitor"], required=True)
    parser.add_argument("--truck-id", default="TRK-001")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    args = parser.parse_args()

    if args.mode == "build-reference":
        build_reference(args.truck_id, args.broker_host, args.broker_port)
    else:
        monitor(args.truck_id, args.broker_host, args.broker_port)


if __name__ == "__main__":
    main()