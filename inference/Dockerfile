# LogiEdge Inference Service
#
# Build from the PROJECT ROOT so both data_pipeline/ and inference/ are
# visible to the build context:
#   docker build -f inference/Dockerfile -t logibridge-inference:latest .
#
# Layer ordering is deliberate: dependency and code layers are installed
# BEFORE the model file is copied in, so that changing only model.tflite
# and rebuilding reuses every cached layer up to that point (this is the
# basis of the OTA layer-cache bandwidth-saving demo).

FROM python:3.11-slim

WORKDIR /app

# ---- dependency layer (cached across model updates) ------------------------
COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- application code layer (cached across model updates) ------------------
COPY data_pipeline/preprocessing.py ./preprocessing.py
COPY inference/inference_service.py .

# ---- variable layers: only these change on a model update ------------------
COPY inference/model.tflite .
COPY data_pipeline/training_stats.npy .

ENV MODEL_PATH=/app/model.tflite
ENV STATS_PATH=/app/training_stats.npy
ENV TRUCK_ID=TRK-001
ENV BROKER_HOST=mosquitto
ENV BROKER_PORT=1883

CMD ["python", "inference_service.py"]
