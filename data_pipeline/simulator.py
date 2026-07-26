"""
LogiEdge Sensor Simulator
--------------------------
Simulates a refrigerated truck's cold-chain sensor payload and publishes it
to a local Mosquitto MQTT broker.

Streams
    temperature   : 1 Hz    topic = logibridge/trucks/{truck_id}/temperature
    vibration_rms : 0.5 Hz  topic = logibridge/trucks/{truck_id}/vibration
    door_event    : event   topic = logibridge/trucks/{truck_id}/door

Usage
    python simulator.py --anomaly none
    python simulator.py --anomaly temp_drift --truck-id TRK-014 --duration 1200
    python simulator.py --anomaly vibration
    python simulator.py --anomaly combined

Requires
    pip install paho-mqtt numpy
    A Mosquitto broker running on localhost:1883
"""

import argparse
import json
import random
import threading
import time
from datetime import datetime, timezone

import numpy as np
import paho.mqtt.client as mqtt

SETPOINT_C = 4.0
TEMP_STD_NORMAL = 0.3
TEMP_DRIFT_PER_READING = 0.08          # deg C added per reading in temp_drift mode

VIB_MEAN_NORMAL = 0.45
VIB_STD_NORMAL = 0.05
VIB_MEAN_ANOMALY = 1.2
VIB_STD_ANOMALY = 0.15

BROKER_HOST = "localhost"
BROKER_PORT = 1883


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


class TruckSensorSimulator:
    def __init__(self, truck_id: str, anomaly_mode: str, broker_host: str, broker_port: int):
        self.truck_id = truck_id
        self.anomaly_mode = anomaly_mode
        self._drift_accum = 0.0
        self._stop_event = threading.Event()

        self.client = mqtt.Client(client_id=f"simulator-{truck_id}",
                                   callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.connect(broker_host, broker_port, keepalive=60)
        self.client.loop_start()

        self.temp_topic = f"logibridge/trucks/{truck_id}/temperature"
        self.vib_topic = f"logibridge/trucks/{truck_id}/vibration"
        self.door_topic = f"logibridge/trucks/{truck_id}/door"

    # ---- signal generators -------------------------------------------------
    def _next_temperature(self) -> float:
        if self.anomaly_mode in ("temp_drift", "combined"):
            self._drift_accum += TEMP_DRIFT_PER_READING
        value = np.random.normal(SETPOINT_C + self._drift_accum, TEMP_STD_NORMAL)
        return round(float(value), 3)

    def _next_vibration(self) -> float:
        if self.anomaly_mode in ("vibration", "combined"):
            value = np.random.normal(VIB_MEAN_ANOMALY, VIB_STD_ANOMALY)
        else:
            value = np.random.normal(VIB_MEAN_NORMAL, VIB_STD_NORMAL)
        return round(max(float(value), 0.0), 4)

    # ---- publish loops -------------------------------------------------
    def _temperature_loop(self):
        while not self._stop_event.is_set():
            payload = {
                "truck_id": self.truck_id,
                "timestamp": utc_now_iso(),
                "temperature_c": self._next_temperature(),
            }
            self.client.publish(self.temp_topic, json.dumps(payload), qos=1)
            time.sleep(1.0)

    def _vibration_loop(self):
        while not self._stop_event.is_set():
            payload = {
                "truck_id": self.truck_id,
                "timestamp": utc_now_iso(),
                "vibration_rms_g": self._next_vibration(),
            }
            self.client.publish(self.vib_topic, json.dumps(payload), qos=1)
            time.sleep(2.0)  # 0.5 Hz

    def _door_loop(self):
        # Sparse, random discrete events roughly every 3-9 minutes
        while not self._stop_event.is_set():
            time.sleep(random.uniform(180, 540))
            for state in ("OPEN", "CLOSE"):
                payload = {
                    "truck_id": self.truck_id,
                    "timestamp": utc_now_iso(),
                    "door_state": state,
                }
                self.client.publish(self.door_topic, json.dumps(payload), qos=1)
                if state == "OPEN":
                    time.sleep(random.uniform(15, 90))

    def run(self, duration_seconds: float):
        threads = [
            threading.Thread(target=self._temperature_loop, daemon=True),
            threading.Thread(target=self._vibration_loop, daemon=True),
            threading.Thread(target=self._door_loop, daemon=True),
        ]
        for t in threads:
            t.start()
        print(f"[simulator] truck_id={self.truck_id} anomaly={self.anomaly_mode} "
              f"duration={duration_seconds}s -- publishing to {BROKER_HOST}:{BROKER_PORT}")
        try:
            time.sleep(duration_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_event.set()
            time.sleep(0.5)
            self.client.loop_stop()
            self.client.disconnect()
            print("[simulator] stopped.")


def main():
    parser = argparse.ArgumentParser(description="LogiEdge cold-chain sensor simulator")
    parser.add_argument("--anomaly", choices=["none", "temp_drift", "vibration", "combined"],
                         default="none")
    parser.add_argument("--truck-id", default="TRK-001")
    parser.add_argument("--duration", type=float, default=300.0,
                         help="Run duration in seconds (default 300)")
    parser.add_argument("--broker-host", default=BROKER_HOST)
    parser.add_argument("--broker-port", type=int, default=BROKER_PORT)
    args = parser.parse_args()

    sim = TruckSensorSimulator(args.truck_id, args.anomaly, args.broker_host, args.broker_port)
    sim.run(args.duration)


if __name__ == "__main__":
    main()
