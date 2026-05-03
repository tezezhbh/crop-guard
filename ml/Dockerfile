# ml/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use PORT env var (default 8000, HuggingFace Spaces uses 7860)
ENV PORT=8000

EXPOSE ${PORT}

CMD uvicorn serve:app --host 0.0.0.0 --port ${PORT}
