# Python ML Service Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model dosyalarını kopyala
COPY ../drone_fault_detection_model /app/drone_fault_detection_model

# ML servisi kopyala
COPY ml_service.py .

# Port
EXPOSE 5000

# Başlat
CMD ["python", "ml_service.py"]
