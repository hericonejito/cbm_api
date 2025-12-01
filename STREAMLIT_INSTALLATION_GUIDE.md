# CBM Streamlit Application - Installation Guide

This guide provides detailed instructions for installing and running the CBM Expert Feedback System Streamlit application.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
  - [Option 1: Local Installation](#option-1-local-installation)
  - [Option 2: Docker Installation](#option-2-docker-installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Updating the Application](#updating-the-application)
- [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **RAM**: Minimum 8 GB (16 GB recommended for video processing)
- **Storage**: At least 10 GB free space (more needed for video storage)
- **CPU**: Multi-core processor recommended (4+ cores)
- **GPU**: Optional, but recommended for faster model inference

### Software Requirements

**For Local Installation:**
- Python 3.10 or higher
- pip (Python package manager)
- Git (for cloning the repository)
- FFmpeg (for video processing)

**For Docker Installation:**
- Docker 20.10+
- Docker Compose 1.29+ (optional but recommended)

---

## Installation Methods

### Option 1: Local Installation

#### Step 1: Install System Dependencies

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip ffmpeg libsm6 libxext6 git
```

**On macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.10 ffmpeg git
```

**On Windows:**
1. Download and install Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
3. Install Git from [git-scm.com](https://git-scm.com/download/win)

#### Step 2: Clone the Repository

```bash
# Navigate to your desired installation directory
cd /path/to/your/projects

# Clone the repository (replace with your actual repo URL)
git clone <repository-url> cbm_api
cd cbm_api
```

If you already have the code, simply navigate to the project directory:
```bash
cd /path/to/cbm_api
```

#### Step 3: Create a Python Virtual Environment (Recommended)

**On Linux/macOS:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### Step 4: Install Python Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Install Streamlit if not already included
pip install streamlit
```

#### Step 5: Set Up Directory Structure

```bash
# Create necessary directories
mkdir -p shared_data/videos_to_frame
mkdir -p shared_data/video_frames
mkdir -p shared_data/predictions
mkdir -p shared_data/active_learning_feedback
mkdir -p shared_data/classes.txt

# Verify directories were created
ls -la shared_data/
```

#### Step 6: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10+

# Check installed packages
pip list | grep streamlit
pip list | grep fastapi
pip list | grep torch

# Verify FFmpeg installation
ffmpeg -version
```

---

### Option 2: Docker Installation

Docker provides an isolated environment and is the easiest way to get started.

#### Step 1: Install Docker

**On Ubuntu:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install docker-compose-plugin

# Add your user to the docker group (optional, to run without sudo)
sudo usermod -aG docker $USER
newgrp docker
```

**On macOS:**
1. Download and install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Start Docker Desktop

**On Windows:**
1. Download and install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Enable WSL 2 if prompted
3. Start Docker Desktop

#### Step 2: Navigate to Project Directory

```bash
cd /path/to/cbm_api
```

#### Step 3: Build the Docker Image

```bash
# Build the Docker image
docker-compose build

# This will take 5-10 minutes on first build
```

#### Step 4: Verify Docker Installation

```bash
# Check Docker is running
docker --version
docker-compose --version

# Verify the image was built
docker images | grep cbm
```

---

## Configuration

### API Configuration

The Streamlit app needs to connect to the FastAPI backend. By default, it connects to `http://localhost:8000`.

#### Local Installation Configuration

Edit the Streamlit configuration in `streamlit/streamlit_ui.py` (line 16):

```python
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
```

Or set the environment variable:

**On Linux/macOS:**
```bash
export API_BASE_URL="http://localhost:8000"
```

**On Windows:**
```bash
set API_BASE_URL=http://localhost:8000
```

#### Docker Configuration

For Docker, the API runs inside a container. If both API and Streamlit are in containers, use the service name:

```bash
export API_BASE_URL="http://cbm-api:8000"
```

### Port Configuration

**Default Ports:**
- FastAPI Backend: `8000`
- Streamlit Frontend: `8501` (default Streamlit port)

To change the Streamlit port:
```bash
streamlit run streamlit/streamlit_ui.py --server.port 8080
```

---

## Running the Application

### Method 1: Local Installation

#### Step 1: Start the FastAPI Backend

In a **first terminal window**, activate your virtual environment and start the API:

```bash
cd /path/to/cbm_api

# Activate virtual environment (if using one)
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate  # On Windows

# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

#### Step 2: Start the Streamlit Frontend

In a **second terminal window**, activate your virtual environment and start Streamlit:

```bash
cd /path/to/cbm_api

# Activate virtual environment (if using one)
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate  # On Windows

# Start Streamlit
streamlit run streamlit/streamlit_ui.py
```

You should see output like:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

#### Step 3: Access the Application

Open your web browser and navigate to:
```
http://localhost:8501
```

---

### Method 2: Docker Installation

#### Option A: Using Docker Compose (API Only)

Start the FastAPI backend:

```bash
cd /path/to/cbm_api

# Start the API container
docker-compose up -d

# Check it's running
docker-compose ps
docker-compose logs
```

Then run Streamlit locally:
```bash
# Set API URL to connect to Docker container
export API_BASE_URL="http://localhost:8000"

# Run Streamlit
streamlit run streamlit/streamlit_ui.py
```

Access the application at: `http://localhost:8501`

#### Option B: Running Streamlit in Docker (Advanced)

To run both API and Streamlit in Docker, you'll need to modify `docker-compose.yml`:

**Edit `docker-compose.yml` to add Streamlit service:**

```yaml
version: '3.8'

services:
  cbm-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./shared_data:/app/shared_data
      - ./app/trained_model:/app/app/trained_model
    environment:
      - PYTHONPATH=/app
      - PYTHONUNBUFFERED=1
    restart: unless-stopped

  cbm-streamlit:
    build: .
    command: streamlit run streamlit/streamlit_ui.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://cbm-api:8000
    depends_on:
      - cbm-api
    restart: unless-stopped
```

Then start both services:
```bash
docker-compose up -d
```

Access the application at: `http://localhost:8501`

---

## Verification

### 1. Check API Connection

Once both services are running, verify the API is accessible:

**Using curl:**
```bash
curl http://localhost:8000/feedback/stats
```

You should receive a JSON response with feedback statistics.

**Using browser:**
Navigate to: `http://localhost:8000/docs` to see the FastAPI interactive documentation.

### 2. Check Streamlit App

1. Open `http://localhost:8501` in your browser
2. You should see the CBM Expert Feedback System dashboard
3. Check the sidebar - the **System Status** should show "✅ API Connected" in green

### 3. Test Basic Functionality

1. Navigate to **🎥 Video Processing** section
2. Try uploading a small test video (if available)
3. Check that the upload completes without errors

---

## Troubleshooting

### Problem: "API Disconnected" Error in Streamlit

**Symptoms:** Red status in sidebar showing "❌ API Disconnected"

**Solutions:**
1. Verify FastAPI is running:
   ```bash
   curl http://localhost:8000/feedback/stats
   ```

2. Check the API_BASE_URL:
   ```bash
   # In Python
   import os
   print(os.getenv("API_BASE_URL", "http://localhost:8000"))
   ```

3. Check for port conflicts:
   ```bash
   # On Linux/macOS
   lsof -i :8000

   # On Windows
   netstat -ano | findstr :8000
   ```

4. Check FastAPI logs for errors

### Problem: ModuleNotFoundError

**Symptoms:** `ModuleNotFoundError: No module named 'streamlit'` or similar

**Solutions:**
1. Ensure virtual environment is activated
2. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt
   pip install streamlit
   ```

3. Verify installation:
   ```bash
   pip list | grep streamlit
   ```

### Problem: FFmpeg Not Found

**Symptoms:** Errors when processing videos: "FFmpeg not found"

**Solutions:**

**On Linux:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**On macOS:**
```bash
brew install ffmpeg
```

**On Windows:**
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to system PATH

**Verify installation:**
```bash
ffmpeg -version
```

### Problem: Port Already in Use

**Symptoms:** "Address already in use" error

**Solutions:**

**For API (port 8000):**
```bash
# Find what's using the port
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Kill the process or use a different port
uvicorn app.main:app --port 8001
```

**For Streamlit (port 8501):**
```bash
streamlit run streamlit/streamlit_ui.py --server.port 8502
```

### Problem: Docker Container Won't Start

**Symptoms:** Container exits immediately or won't build

**Solutions:**

1. Check Docker logs:
   ```bash
   docker-compose logs cbm-api
   ```

2. Rebuild without cache:
   ```bash
   docker-compose build --no-cache
   docker-compose up
   ```

3. Check disk space:
   ```bash
   df -h  # Linux/macOS
   ```

4. Verify Docker is running:
   ```bash
   docker ps
   ```

### Problem: Slow Video Processing

**Symptoms:** Video upload or frame extraction takes very long

**Solutions:**

1. **Increase frame interval**: Use higher frame interval (e.g., 50 instead of 10)
2. **Reduce video size**: Compress videos before uploading
3. **Check system resources**:
   ```bash
   top  # Linux/macOS
   # or
   htop
   ```
4. **Use GPU acceleration** if available (requires PyTorch with CUDA)

### Problem: Model Predictions Not Working

**Symptoms:** No predictions generated or errors during prediction

**Solutions:**

1. Verify model files exist:
   ```bash
   ls -la app/trained_model/
   ```

   Should contain:
   - `W_c.pt`
   - `W_g.pt`
   - `b_g.pt`
   - `proj_mean.pt`
   - `proj_std.pt`
   - `args.txt`
   - `concepts.txt`

2. Check model file permissions:
   ```bash
   chmod -R 755 app/trained_model/
   ```

3. Review API logs for detailed error messages

### Problem: Out of Memory Errors

**Symptoms:** Application crashes with "Out of memory" or similar

**Solutions:**

1. **Reduce batch size**: Lower the frame extraction rate
2. **Close other applications**
3. **Increase system swap space** (Linux)
4. **Use Docker memory limits**:
   ```yaml
   services:
     cbm-api:
       mem_limit: 4g
   ```

---

## Updating the Application

### Local Installation

```bash
cd /path/to/cbm_api

# Pull latest changes (if using git)
git pull origin main

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows

# Update dependencies
pip install --upgrade -r requirements.txt

# Restart the application
```

### Docker Installation

```bash
cd /path/to/cbm_api

# Pull latest changes
git pull origin main

# Rebuild containers
docker-compose build

# Restart services
docker-compose down
docker-compose up -d
```

---

## Advanced Configuration

### Custom Environment Variables

Create a `.env` file in the project root:

```bash
# .env file
API_BASE_URL=http://localhost:8000
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
PYTHONPATH=/app
```

Load it when running:

**For Streamlit:**
```bash
export $(cat .env | xargs)
streamlit run streamlit/streamlit_ui.py
```

**For Docker:**
```yaml
# docker-compose.yml
services:
  cbm-api:
    env_file:
      - .env
```

### Running in Production

For production deployments:

1. **Disable debug mode** in FastAPI:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000  # Remove --reload
   ```

2. **Use a process manager** like systemd or supervisor

3. **Set up reverse proxy** with Nginx or Apache

4. **Enable HTTPS** with SSL certificates

5. **Configure firewall rules**

6. **Set up monitoring** and logging

7. **Regular backups** of `shared_data/` directory

---

## Directory Structure Overview

After installation, your directory structure should look like:

```
cbm_api/
├── app/                              # FastAPI application
│   ├── main.py                       # API entry point
│   ├── cbm_model.py                  # Model definitions
│   ├── intervention.py               # Prediction logic
│   ├── feedback_manager.py           # Feedback handling
│   └── trained_model/                # Pre-trained model weights
│       ├── W_c.pt
│       ├── W_g.pt
│       ├── args.txt
│       └── ...
├── streamlit/                         # Streamlit application
│   └── streamlit_ui.py               # Main UI file
├── shared_data/                       # Runtime data (created on first run)
│   ├── videos_to_frame/              # Uploaded videos
│   ├── video_frames/                 # Extracted frames
│   ├── predictions/                  # Model predictions
│   └── active_learning_feedback/     # Expert feedback
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker configuration
├── docker-compose.yml                 # Docker Compose config
├── README.md                          # Basic readme
├── STREAMLIT_INSTALLATION_GUIDE.md   # This file
└── STREAMLIT_USER_GUIDE.md          # User documentation
```

---

## Next Steps

After successful installation, refer to the **[STREAMLIT_USER_GUIDE.md](STREAMLIT_USER_GUIDE.md)** for:

- How to use the application
- Dashboard overview
- Video processing workflow
- Expert feedback submission
- Model retraining process
- Best practices and tips

---

## Getting Help

### Common Resources

- **API Documentation**: `http://localhost:8000/docs` (when API is running)
- **User Guide**: See `STREAMLIT_USER_GUIDE.md`
- **System Logs**: Check terminal output for both API and Streamlit

### Checking System Status

**API Health Check:**
```bash
curl http://localhost:8000/feedback/stats
```

**Check Running Processes:**
```bash
# Local installation
ps aux | grep uvicorn
ps aux | grep streamlit

# Docker installation
docker-compose ps
docker-compose logs
```

**View Real-time Logs:**
```bash
# Docker
docker-compose logs -f

# Local - check terminal windows where services are running
```

---

## Uninstalling

### Local Installation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv/

# Remove data (optional - WARNING: deletes all data)
rm -rf shared_data/

# Remove entire installation (optional)
cd ..
rm -rf cbm_api/
```

### Docker Installation

```bash
# Stop and remove containers
docker-compose down

# Remove Docker images
docker rmi cbm_api-cbm-api

# Remove data volumes (optional - WARNING: deletes all data)
rm -rf shared_data/
```

---

**Version:** 1.0
**Last Updated:** 2025
**Compatibility:** CBM API v2.0, Streamlit 1.x+, Python 3.10+
