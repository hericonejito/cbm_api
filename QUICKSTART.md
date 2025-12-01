# CBM Streamlit App - Quick Start Cheat Sheet

**One-page reference for getting the CBM Streamlit application up and running quickly.**

---

## Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] FFmpeg installed
- [ ] Docker installed (if using Docker method)
- [ ] At least 8 GB RAM available
- [ ] 10+ GB free disk space

---

## Installation (Choose One Method)

### Method 1: Local Installation (5 minutes)

```bash
# 1. Navigate to project directory
cd /path/to/cbm_api

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit

# 4. Create data directories
mkdir -p shared_data/{videos_to_frame,video_frames,predictions,active_learning_feedback}
```

### Method 2: Docker Installation (10 minutes)

```bash
# 1. Navigate to project directory
cd /path/to/cbm_api

# 2. Build and start containers
docker-compose up -d

# 3. Verify it's running
docker-compose ps
docker-compose logs
```

---

## Running the Application

### Local Installation

**Terminal 1 - Start API:**
```bash
cd /path/to/cbm_api
source venv/bin/activate  # Activate venv
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start Streamlit:**
```bash
cd /path/to/cbm_api
source venv/bin/activate  # Activate venv
streamlit run streamlit/streamlit_ui.py
```

**Access:**
- Streamlit UI: `http://localhost:8501`
- API Docs: `http://localhost:8000/docs`

### Docker Installation

**Start:**
```bash
docker-compose up -d
```

**For Streamlit (run locally):**
```bash
export API_BASE_URL="http://localhost:8000"
streamlit run streamlit/streamlit_ui.py
```

**Access:**
- Streamlit UI: `http://localhost:8501`
- API: `http://localhost:8000`

---

## Common Commands

### Local Installation

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate      # Windows

# Start API (Terminal 1)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit (Terminal 2)
streamlit run streamlit/streamlit_ui.py

# Stop: Press CTRL+C in each terminal

# Check API status
curl http://localhost:8000/feedback/stats
```

### Docker Installation

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart services
docker-compose restart

# Rebuild after changes
docker-compose build --no-cache
docker-compose up -d

# Check status
docker-compose ps
```

---

## Troubleshooting Quick Fixes

### "API Disconnected" in Streamlit
```bash
# Check if API is running
curl http://localhost:8000/feedback/stats

# Restart API (local)
# Press CTRL+C in Terminal 1, then:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Restart API (Docker)
docker-compose restart cbm-api
```

### Port Already in Use
```bash
# Use different ports
uvicorn app.main:app --port 8001  # API on 8001
streamlit run streamlit/streamlit_ui.py --server.port 8502  # Streamlit on 8502
```

### Module Not Found
```bash
# Ensure venv is activated, then:
pip install -r requirements.txt
pip install streamlit
```

### FFmpeg Not Found
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows: Download from ffmpeg.org and add to PATH
```

### Out of Memory
- Reduce frame interval when uploading videos
- Close other applications
- Use smaller video files for testing

---

## First-Time Workflow

1. **Start both services** (API + Streamlit)
2. **Open browser**: `http://localhost:8501`
3. **Check status**: Sidebar should show "✅ API Connected"
4. **Upload test video**:
   - Go to "🎥 Video Processing"
   - Upload a small video file
   - Set frame interval to 30
   - Enable "Auto-run predictions"
   - Click "Upload & Process"
5. **Review predictions**:
   - Go to "📝 Expert Feedback"
   - Select your video
   - Review and submit feedback
6. **Continue until 50+ samples**
7. **Retrain model**:
   - Go to "🤖 Model Management"
   - Click "Start Retraining"

---

## Key URLs

| Service | URL | Description |
|---------|-----|-------------|
| Streamlit UI | `http://localhost:8501` | Main web interface |
| API Docs | `http://localhost:8000/docs` | Interactive API documentation |
| API Health | `http://localhost:8000/feedback/stats` | Quick health check |

---

## Environment Variables

```bash
# Set API URL (if needed)
export API_BASE_URL="http://localhost:8000"  # Linux/macOS
set API_BASE_URL=http://localhost:8000       # Windows

# Change Streamlit port
export STREAMLIT_SERVER_PORT=8502
```

---

## Stopping the Application

### Local Installation
- Press **CTRL+C** in each terminal window (API and Streamlit)

### Docker Installation
```bash
docker-compose down
```

---

## Directory Structure (Reference)

```
cbm_api/
├── app/                    # FastAPI backend
│   ├── main.py            # API entry point
│   └── trained_model/     # Model weights (required)
├── streamlit/             # Streamlit frontend
│   └── streamlit_ui.py   # UI entry point
├── shared_data/           # Runtime data (created automatically)
│   ├── videos_to_frame/
│   ├── video_frames/
│   ├── predictions/
│   └── active_learning_feedback/
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker config
└── README.md             # Main documentation
```

---

## Useful Tips

1. **Test with small videos first** (< 50 MB)
2. **Frame interval of 30** is a good starting point
3. **Sort by "Confidence (Low to High)"** to focus feedback on uncertain predictions
4. **Both corrections AND confirmations** help improve the model
5. **Wait for 50+ samples** before retraining
6. **Backup data regularly**: Copy `shared_data/` folder

---

## Getting More Help

- **Installation Details**: See `STREAMLIT_INSTALLATION_GUIDE.md`
- **Usage Guide**: See `STREAMLIT_USER_GUIDE.md`
- **API Documentation**: Visit `http://localhost:8000/docs` when API is running

---

## System Requirements Quick Check

```bash
# Check Python version (need 3.10+)
python --version

# Check FFmpeg (need for video processing)
ffmpeg -version

# Check Docker (if using Docker method)
docker --version
docker-compose --version

# Check disk space (need 10+ GB)
df -h  # Linux/macOS
```

---

## Quick Update Commands

### Local Installation
```bash
cd /path/to/cbm_api
git pull origin main  # If using git
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Docker Installation
```bash
cd /path/to/cbm_api
git pull origin main  # If using git
docker-compose build
docker-compose down
docker-compose up -d
```

---

**Keep this cheat sheet handy for quick reference!**

For comprehensive guides, see:
- [STREAMLIT_INSTALLATION_GUIDE.md](STREAMLIT_INSTALLATION_GUIDE.md)
- [STREAMLIT_USER_GUIDE.md](STREAMLIT_USER_GUIDE.md)
