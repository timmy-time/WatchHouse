FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV PYTHONPATH=/app
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
ENV FFMPEG_BIN=/usr/bin/ffmpeg
ENV FFPROBE_BIN=/usr/bin/ffprobe
ENV FACE_MODEL_DIR=/opt/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n weights into the container image
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Pre-download face models into /opt/models
RUN python3 -c "import os, urllib.request; os.makedirs('/opt/models', exist_ok=True); \
urllib.request.urlretrieve('https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx', '/opt/models/face_detection_yunet_2023mar.onnx'); \
urllib.request.urlretrieve('https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx', '/opt/models/face_recognition_sface_2021dec.onnx')"

COPY . .

EXPOSE 8080
ENTRYPOINT ["python3", "main.py"]
CMD ["scan", "--input", "/data/clips", "--output", "/data/output", "--gpus", "0,1"]
