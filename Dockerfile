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

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]