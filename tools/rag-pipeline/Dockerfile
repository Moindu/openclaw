FROM python:3.12-slim

# Install system dependencies for OCR/video pipeline
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .

# Create book directories
RUN mkdir -p /data/books/incoming /data/books/originals /data/books/extracted \
    /data/books/processed /data/books/failed

# Run both watcher and API server
CMD ["sh", "-c", "python watcher.py & python server.py"]
