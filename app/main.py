import json
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.cbm_model import CBM_model, load_cbm
from app.feedback_manager import save_feedback
from app.retrain import retrain_cbm
from app.video_processor import process_video
from app.intervention import load_model_and_data, predict_and_visualize
import numpy as np

app = FastAPI()

# ✅ Get absolute path to shared_data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DATA_DIR = os.path.join(BASE_DIR, "..", "shared_data")
SHARED_DATA_DIR = os.path.abspath(SHARED_DATA_DIR)

TRAINED_MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
print('Trained Model Dir ' + TRAINED_MODEL_DIR)

# Active Learning Configuration
FEEDBACK_DATA_DIR = os.path.join(SHARED_DATA_DIR, "active_learning_feedback")
ANNOTATIONS_FILE = "annotations.json"
RETRAIN_TRIGGER_COUNT = 50  # Retrain after N new samples (adjusted for your use case)

# ✅ Mount using absolute path
app.mount("/shared_data", StaticFiles(directory=SHARED_DATA_DIR), name="shared_data")
model = load_cbm(TRAINED_MODEL_DIR, 'cpu')


# Pydantic models for Active Learning
class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: str
    samples_collected: int
    ready_for_retrain: bool


def to_url_path(abs_path):
    """Convert absolute shared data path to a URL that can be accessed via FastAPI static mount."""
    if SHARED_DATA_DIR in abs_path:
        relative_path = os.path.relpath(abs_path, SHARED_DATA_DIR)
        return f"/shared_data/{relative_path.replace(os.sep, '/')}"
    else:
        raise ValueError("Path not inside SHARED_DATA_DIR")


def ensure_feedback_directories():
    """Create necessary directories for storing active learning feedback data"""
    base_path = Path(FEEDBACK_DATA_DIR)
    base_path.mkdir(exist_ok=True)

    # Create subdirectories for different outlier types (adjust based on your classes)
    outlier_types = ["normal", "crack", "corrosion", "leakage"]
    for outlier_type in outlier_types:
        (base_path / outlier_type).mkdir(exist_ok=True)

    return base_path


def load_annotations():
    """Load existing annotations from JSON file"""
    annotations_path = Path(FEEDBACK_DATA_DIR) / ANNOTATIONS_FILE
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            return json.load(f)
    return []


def save_annotations(annotations):
    """Save annotations to JSON file"""
    annotations_path = Path(FEEDBACK_DATA_DIR) / ANNOTATIONS_FILE
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=2)


def get_sample_count():
    """Count total samples collected"""
    annotations = load_annotations()
    return len(annotations)


@app.post("/predict_frames")
async def predict_frames(request: dict):
    video_name = "multi_train"
    model_loaded, val_data_t, val_pil_data, classes, concepts, dataset = load_model_and_data(TRAINED_MODEL_DIR, 'cpu')
    frame_folder = os.path.join(SHARED_DATA_DIR, video_name)
    try:
        predictions = predict_and_visualize(model_loaded, val_data_t, val_pil_data, classes, concepts, dataset,
                                            'cpu', np.linspace(0, len(val_data_t) - 1, len(val_data_t), dtype=int))

        formatted_results = [
            {
                "path": prediction['filename'],
                "label": prediction['class'],
                "confidence": prediction['confidence'],
                "frame_id": f"frame_{i}_{uuid.uuid4().hex[:8]}"  # Add unique frame ID for feedback tracking
            } for i, prediction in enumerate(predictions)
        ]
        print(formatted_results)
        return JSONResponse(content={"results": json.dumps(formatted_results)}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Enhanced feedback endpoint for active learning
@app.post("/feedback/expert", response_model=FeedbackResponse)
async def submit_expert_feedback(
        frame_image: UploadFile = File(...),
        frame_id: str = Form(...),
        model_prediction: str = Form(...),
        expert_classification: str = Form(...),
        confidence_score: Optional[float] = Form(None),
        expert_notes: Optional[str] = Form(None),
        pipe_id: Optional[str] = Form(None)
):
    """
    Submit expert feedback for active learning

    Enhanced version of the original feedback endpoint with proper data organization
    for retraining the CBM model.
    """
    try:
        # Ensure directories exist
        base_path = ensure_feedback_directories()

        # Generate unique feedback ID
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Validate expert classification (adjust these based on your actual classes)
        valid_classifications = ["normal", "crack", "corrosion", "leakage"]
        if expert_classification.lower() not in valid_classifications:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid classification. Must be one of: {valid_classifications}"
            )

        # Save image to appropriate folder
        classification_folder = base_path / expert_classification.lower()
        image_filename = f"{feedback_id}_{frame_id}.jpg"
        image_path = classification_folder / image_filename

        # Read and save the uploaded image
        contents = await frame_image.read()
        with open(image_path, "wb") as f:
            f.write(contents)

        # Create annotation entry
        annotation = {
            "feedback_id": feedback_id,
            "frame_id": frame_id,
            "image_path": str(image_path),
            "model_prediction": model_prediction,
            "expert_classification": expert_classification.lower(),
            "confidence_score": confidence_score,
            "expert_notes": expert_notes,
            "pipe_id": pipe_id,
            "timestamp": timestamp,
            "is_correction": model_prediction.lower() != expert_classification.lower()
        }

        # Load existing annotations and add new one
        annotations = load_annotations()
        annotations.append(annotation)
        save_annotations(annotations)

        # Also save to your existing feedback system for compatibility
        feedback_data = {
            "image_id": frame_id,
            "true_label": expert_classification,
            "explanation": expert_notes or f"Expert correction from {model_prediction} to {expert_classification}"
        }
        save_feedback(feedback_data)

        # Check if ready for retraining
        samples_collected = len(annotations)
        ready_for_retrain = samples_collected >= RETRAIN_TRIGGER_COUNT

        return FeedbackResponse(
            status="success",
            message=f"Expert feedback recorded successfully. Image saved to {classification_folder.name} folder.",
            feedback_id=feedback_id,
            samples_collected=samples_collected,
            ready_for_retrain=ready_for_retrain
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing expert feedback: {str(e)}")


# Keep your original feedback endpoint for backward compatibility
@app.post("/feedback")
async def feedback(image_id: str = Form(...), true_label: str = Form(...), explanation: str = Form(...)):
    feedback_data = {
        "image_id": image_id,
        "true_label": true_label,
        "explanation": explanation
    }
    save_feedback(feedback_data)
    return {"status": "Feedback received"}


@app.get("/feedback/stats")
async def get_feedback_stats():
    """Get statistics about collected expert feedback for active learning"""
    try:
        annotations = load_annotations()

        if not annotations:
            return {
                "total_samples": 0,
                "by_classification": {},
                "corrections_count": 0,
                "ready_for_retrain": False
            }

        # Count by classification
        classification_counts = {}
        corrections_count = 0

        for annotation in annotations:
            cls = annotation["expert_classification"]
            classification_counts[cls] = classification_counts.get(cls, 0) + 1

            if annotation.get("is_correction", False):
                corrections_count += 1

        return {
            "total_samples": len(annotations),
            "by_classification": classification_counts,
            "corrections_count": corrections_count,
            "correction_rate": corrections_count / len(annotations) if annotations else 0,
            "ready_for_retrain": len(annotations) >= RETRAIN_TRIGGER_COUNT,
            "retrain_threshold": RETRAIN_TRIGGER_COUNT
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feedback stats: {str(e)}")


@app.post("/feedback/prepare_retrain")
async def prepare_for_retraining():
    """Prepare active learning data for CBM model retraining"""
    try:
        annotations = load_annotations()

        if len(annotations) < RETRAIN_TRIGGER_COUNT:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough samples for retraining. Need {RETRAIN_TRIGGER_COUNT}, have {len(annotations)}"
            )

        # Create training data structure
        training_data = {
            "annotations": annotations,
            "data_directory": FEEDBACK_DATA_DIR,
            "total_samples": len(annotations),
            "created_at": datetime.now().isoformat(),
            "class_distribution": {}
        }

        # Calculate class distribution
        for annotation in annotations:
            cls = annotation["expert_classification"]
            training_data["class_distribution"][cls] = training_data["class_distribution"].get(cls, 0) + 1

        # Save training configuration
        training_config_path = Path(FEEDBACK_DATA_DIR) / "training_config.json"
        with open(training_config_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        return {
            "status": "ready",
            "message": "Active learning data prepared for retraining",
            "config_file": str(training_config_path),
            "total_samples": len(annotations),
            "class_distribution": training_data["class_distribution"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing for retraining: {str(e)}")


@app.post("/retrain")
async def retrain():
    """Enhanced retrain endpoint that can use active learning data"""
    try:
        # Check if we have active learning data ready
        annotations = load_annotations()

        if len(annotations) >= RETRAIN_TRIGGER_COUNT:
            # Use active learning data for retraining
            print(f"Retraining with {len(annotations)} active learning samples")
            retrain_cbm(FEEDBACK_DATA_DIR, "static/images", "feedback")
        else:
            # Fall back to original retraining method
            print("Using original retraining method")
            retrain_cbm("data", "static/images", "feedback")

        # Reload the model
        global model
        model = load_cbm(TRAINED_MODEL_DIR, 'cpu')

        return {
            "status": "Model retrained successfully",
            "active_learning_samples": len(annotations),
            "used_active_learning": len(annotations) >= RETRAIN_TRIGGER_COUNT
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query_uncertain")
async def query_uncertain(n: int = 5):
    paths = [os.path.join("sample_pool", p) for p in os.listdir("sample_pool") if p.endswith(".png")]
    return model.rank_uncertain_samples(paths, top_n=n)


# Video processing endpoints (unchanged)
VIDEO_UPLOAD_FOLDER = "shared_data/videos_to_frame"
PROCESSED_OUTPUT_FOLDER = "shared_data/multi_train/Normal"

os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_OUTPUT_FOLDER, exist_ok=True)


@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    """
    Accepts a video file, saves it, processes it using Grad-CAM + CBM logic, and returns results.
    """
    video_path = os.path.join(VIDEO_UPLOAD_FOLDER, file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_name = os.path.splitext(file.filename)[0]

    try:
        results = process_video(video_path, PROCESSED_OUTPUT_FOLDER, video_name, frame_interval=30)
        print(results)

        # Format results for JSON with frame IDs for potential feedback
        formatted_results = [
            {
                "frame_path": path,
                "frame_id": f"video_{video_name}_frame_{i}_{uuid.uuid4().hex[:8]}"
            } for i, path in enumerate(results)
        ]

        return JSONResponse(content={"results": formatted_results}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Utility endpoints for active learning management
@app.get("/feedback/{feedback_id}")
async def get_feedback_details(feedback_id: str):
    """Get details of a specific feedback entry"""
    try:
        annotations = load_annotations()

        for annotation in annotations:
            if annotation["feedback_id"] == feedback_id:
                return annotation

        raise HTTPException(status_code=404, detail="Feedback not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving feedback: {str(e)}")


@app.delete("/feedback/reset")
async def reset_feedback_data():
    """Reset all active learning feedback data (use with caution)"""
    try:
        base_path = Path(FEEDBACK_DATA_DIR)
        if base_path.exists():
            shutil.rmtree(base_path)

        ensure_feedback_directories()

        return {
            "status": "success",
            "message": "All active learning feedback data has been reset"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting feedback data: {str(e)}")


# Initialize directories on startup
@app.on_event("startup")
async def startup_event():
    ensure_feedback_directories()
    print(f"Active Learning Feedback System initialized")
    print(f"Feedback data directory: {FEEDBACK_DATA_DIR}")
    print(f"Current sample count: {get_sample_count()}")
    print(f"Retrain threshold: {RETRAIN_TRIGGER_COUNT}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)