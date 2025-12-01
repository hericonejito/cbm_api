# CBM API - Concept-Based Model for Infrastructure Defect Detection

AI-powered infrastructure inspection system that analyzes video footage to detect and classify defects (cracks, corrosion, leakage) using interpretable machine learning with active learning capabilities.

## Overview

The CBM (Concept-Based Model) system provides:
- **Automated Video Analysis**: Extract frames and detect infrastructure defects
- **Interpretable AI**: 29 human-interpretable concepts explain predictions
- **Active Learning**: Model improves continuously from expert feedback
- **Complete Web Interface**: Streamlit dashboard for video processing, feedback, and model management
- **REST API**: FastAPI backend for programmatic access

**Defect Categories:**
- Normal (no defects)
- Cracks (surface fractures)
- Corrosion (rust, oxidation, material degradation)
- Leakage (water damage, moisture)

## Quick Start

### Running the API (Backend)

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### Running the Streamlit App (Frontend)

```bash
# Start the Streamlit dashboard
streamlit run streamlit/streamlit_ui.py
```

Streamlit app will be available at: `http://localhost:8501`

### Docker Deployment

```bash
# Start services with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Documentation

- **[STREAMLIT_INSTALLATION_GUIDE.md](STREAMLIT_INSTALLATION_GUIDE.md)** - Complete installation instructions
  - Prerequisites and system requirements
  - Local and Docker installation options
  - Configuration and setup
  - Troubleshooting guide

- **[STREAMLIT_USER_GUIDE.md](STREAMLIT_USER_GUIDE.md)** - Application usage guide
  - Dashboard overview
  - Video processing workflow
  - Expert feedback submission
  - Model retraining process
  - Best practices and tips

## Architecture

- **Backend**: FastAPI REST API
- **ML/AI**: PyTorch, CLIP (ViT-B/16), ResNet18-CUB
- **Video Processing**: OpenCV
- **Frontend**: Streamlit web UI
- **Deployment**: Docker + Docker Compose

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload_video` | POST | Upload video and extract frames |
| `/predict_frames/{video_name}` | POST | Run AI predictions on frames |
| `/predictions/{video_name}` | GET | Retrieve predictions with filtering |
| `/feedback/expert` | POST | Submit expert feedback |
| `/retrain` | POST | Retrain model with feedback |
| `/feedback/stats` | GET | Get feedback statistics |
| `/videos` | GET | List all processed videos |

## Project Structure

```
cbm_api/
├── app/                              # FastAPI application
│   ├── main.py                       # API entry point
│   ├── cbm_model.py                  # CBM model architecture
│   ├── intervention.py               # Prediction logic
│   ├── feedback_manager.py           # Feedback handling
│   ├── retrain.py                    # Active learning pipeline
│   └── trained_model/                # Pre-trained model weights
├── streamlit/                         # Streamlit web interface
│   └── streamlit_ui.py               # Main dashboard
├── shared_data/                       # Runtime data storage
│   ├── videos_to_frame/              # Uploaded videos
│   ├── video_frames/                 # Extracted frames
│   ├── predictions/                  # Model predictions
│   └── active_learning_feedback/     # Expert feedback
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker configuration
└── docker-compose.yml                 # Docker Compose config
```

## Requirements

- Python 3.10+
- FFmpeg (for video processing)
- 8+ GB RAM (16 GB recommended)
- CUDA-capable GPU (optional, for faster inference)

## License

[Specify your license here]

## Support

For installation help, see [STREAMLIT_INSTALLATION_GUIDE.md](STREAMLIT_INSTALLATION_GUIDE.md)
For usage help, see [STREAMLIT_USER_GUIDE.md](STREAMLIT_USER_GUIDE.md)
