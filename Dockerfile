# Use Python 3.9 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for video processing and ML libraries
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/shared_data/videos_to_frame \
    /app/shared_data/video_frames \
    /app/shared_data/predictions \
    /app/shared_data/active_learning_feedback \
    /app/trained_model

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# JWT Secret Key - CHANGE THIS IN PRODUCTION!
# Generate with: openssl rand -hex 32
ENV JWT_SECRET_KEY="change-this-secret-key-in-production-use-openssl-rand-hex-32"

# SSL Certificate paths (optional - for HTTPS)
ENV SSL_KEYFILE=/app/certs/key.pem
ENV SSL_CERTFILE=/app/certs/cert.pem

# Create certs directory
RUN mkdir -p /app/certs

# Expose the port the app runs on (8000 for HTTP, 8443 for HTTPS)
EXPOSE 8000 8443

# Command to run the application with HTTPS if certs exist
CMD ["sh", "-c", "if [ -f /app/certs/key.pem ] && [ -f /app/certs/cert.pem ]; then uvicorn app.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile=/app/certs/key.pem --ssl-certfile=/app/certs/cert.pem; else uvicorn app.main:app --host 0.0.0.0 --port 8000; fi"]