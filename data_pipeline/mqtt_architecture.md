# LogiEdge MQTT Architecture

## Topic Tree

```
logibridge/
└── trucks/
    └── {truck_id}/
        ├── temperature   (1 Hz,   QoS 1)
        ├── vibration     (0.5 Hz, QoS 1)
        ├── door          (event,  QoS 1)
        └── inference     (result, QoS 1)
```

## QoS Justification

| Topic | QoS | Reason |
|---|---|---|
| `temperature` | 1 | High frequency, safety-relevant. At-least-once avoids silent loss of a drift reading without doubling storage cost the way QoS 2 would. |
| `vibration` | 1 | Same rationale as temperature; occasional duplicate windows have negligible effect after moving-average filtering. |
| `door` | 1 | Low-frequency discrete events; a dropped event (QoS 0) could hide a chain-of-custody breach, so at-least-once delivery is required. QoS 2 is unnecessary because a duplicate OPEN/CLOSE event does not corrupt the audit trail (both are timestamped). |
| `inference` | 1 | This is the alert-carrying channel to the operations centre relay. Losing an alert is unacceptable; QoS 2's four-way handshake adds latency the 90-second SLA cannot spend on broker overhead. |

## Broker Placement

The Mosquitto broker runs **locally on the truck's edge node** (loopback or on-vehicle LAN), not in the cloud. All classification happens against this local broker so the pipeline works with zero cellular connectivity. A lightweight bridge process (or Mosquitto's native bridge configuration) forwards only the `inference` topic to the operations-centre broker whenever a cellular connection is available, and buffers messages locally when it is not.
