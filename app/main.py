import json
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import your existing modules
from app.cbm_model import CBM_model, load_cbm
from app.feedback_manager import save_feedback
from app.retrain import retrain_cbm
from app.video_processor import process_video
from app.intervention import load_model_and_data, predict_and_visualize
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Right after creating the FastAPI app instance
app = FastAPI(title="CBM Video Processing API", version="1.0.0")

# Add this CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# ✅ Directory Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DATA_DIR = os.path.join(BASE_DIR, "..", "shared_data")
SHARED_DATA_DIR = os.path.abspath(SHARED_DATA_DIR)

# Video and processing directories
VIDEO_UPLOAD_DIR = os.path.join(SHARED_DATA_DIR, "videos_to_frame")
FRAMES_BASE_DIR = os.path.join(SHARED_DATA_DIR, "video_frames")
PREDICTIONS_BASE_DIR = os.path.join(SHARED_DATA_DIR, "predictions")

# Model directories
TRAINED_MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
FEEDBACK_DATA_DIR = os.path.join(SHARED_DATA_DIR, "active_learning_feedback")

# Configuration
RETRAIN_TRIGGER_COUNT = 50
ANNOTATIONS_FILE = "annotations.json"

# Create necessary directories
os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_BASE_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_BASE_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DATA_DIR, exist_ok=True)

# Mount static files
app.mount("/shared_data", StaticFiles(directory=SHARED_DATA_DIR), name="shared_data")

# Load initial model
model = load_cbm(TRAINED_MODEL_DIR, 'cpu')


# Pydantic Models
class VideoUploadResponse(BaseModel):
    status: str
    message: str
    video_id: str
    video_name: str
    upload_path: str
    total_frames_extracted: int


class PredictionRequest(BaseModel):
    video_name: str
    frame_interval: Optional[int] = 50


class PredictionResponse(BaseModel):
    status: str
    video_name: str
    total_predictions: int
    predictions_by_class: Dict[str, int]
    predictions: List[Dict]


class FeedbackRequest(BaseModel):
    video_name: str
    frame_id: str
    model_prediction: str
    expert_classification: str
    confidence_score: Optional[float] = None
    expert_notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: str
    samples_collected: int
    ready_for_retrain: bool


# Utility Functions
def ensure_directories():
    """Create necessary directories for the application"""
    directories = [
        VIDEO_UPLOAD_DIR,
        FRAMES_BASE_DIR,
        PREDICTIONS_BASE_DIR,
        FEEDBACK_DATA_DIR
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Create class-specific feedback directories
    class_names = ["normal", "crack", "corrosion", "leakage"]
    for class_name in class_names:
        Path(FEEDBACK_DATA_DIR, class_name).mkdir(exist_ok=True)


def extract_frames_from_video(video_path: str, output_folder: str, video_name: str, frame_interval: int = 50):
    """Extract frames from video using your existing video_processor"""
    try:
        # Create video-specific frame directory
        video_frame_dir = os.path.join(output_folder, video_name,'Normal')
        os.makedirs(video_frame_dir, exist_ok=True)

        # Use your existing process_video function
        results = process_video(video_path, video_frame_dir, video_name, frame_interval)

        return {
            "success": True,
            "frame_directory": video_frame_dir,
            "total_frames": len(results),
            "frame_paths": results
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


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


# API Endpoints

@app.post("/upload_video", response_model=VideoUploadResponse)
async def upload_video(
        file: UploadFile = File(...),
        frame_interval: int = Form(50, description="Extract every Nth frame")
):
    """
    Upload a video file and extract frames based on the specified interval.

    Args:
        file: Video file to upload
        frame_interval: Extract every Nth frame (default: 50)

    Returns:
        VideoUploadResponse with upload details and frame extraction results
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid video file format")

        # Generate unique video ID and extract video name
        video_id = str(uuid.uuid4())
        video_name = os.path.splitext(file.filename)[0]
        video_filename = f"{video_name}_{video_id}.mp4"

        # Save uploaded video
        video_save_path = os.path.join(VIDEO_UPLOAD_DIR, video_filename)
        with open(video_save_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Extract frames
        frame_extraction_result = extract_frames_from_video(
            video_save_path,
            FRAMES_BASE_DIR,
            f"{video_name}_{video_id}",
            frame_interval
        )

        if not frame_extraction_result["success"]:
            raise HTTPException(status_code=500, detail=f"Frame extraction failed: {frame_extraction_result['error']}")

        return VideoUploadResponse(
            status="success",
            message=f"Video uploaded and {frame_extraction_result['total_frames']} frames extracted successfully",
            video_id=video_id,
            video_name=f"{video_name}_{video_id}",
            upload_path=video_save_path,
            total_frames_extracted=frame_extraction_result['total_frames']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.post("/predict_frames/{video_name}", response_model=PredictionResponse)
async def predict_video_frames(video_name: str):
    """
    Run CBM model predictions on extracted frames for a specific video.

    Args:
        video_name: Name of the video (from upload response)

    Returns:
        PredictionResponse with classification results organized by class
    """
    print(video_name)
    try:
        # Check if video frames exist
        video_frame_dir = os.path.join(FRAMES_BASE_DIR, video_name,'Normal')
        if not os.path.exists(video_frame_dir):
            raise HTTPException(status_code=404, detail=f"No frames found for video: {video_name}")

        # Load model and data
        model_loaded, val_data_t, val_pil_data, classes, concepts, dataset = load_model_and_data(TRAINED_MODEL_DIR,
                                                                                                 'cpu',dataset_name = video_name)

        # Create prediction output directory for this video
        video_prediction_dir = os.path.join(PREDICTIONS_BASE_DIR, video_name)
        os.makedirs(video_prediction_dir, exist_ok=True)

        # Get frame indices (assuming you want to predict all frames)
        frame_files = [f for f in os.listdir(video_frame_dir) if f.endswith(('.jpg', '.png'))]
        frame_indices = list(range(len(frame_files)))

        # Run predictions using your existing function
        predictions = predict_and_visualize(
            model_loaded, val_data_t, val_pil_data, classes, concepts,
            video_prediction_dir, 'cpu', frame_indices
        )

        # Process and organize predictions
        predictions_by_class = {}
        formatted_predictions = []

        for i, prediction in enumerate(predictions):
            class_name = prediction['class']
            predictions_by_class[class_name] = predictions_by_class.get(class_name, 0) + 1

            # Add frame_id for feedback tracking
            frame_id = f"{video_name}_frame_{i}_{uuid.uuid4().hex[:8]}"

            formatted_prediction = {
                "frame_id": frame_id,
                "video_name": video_name,
                "features":prediction['features'],
                "values":prediction['values'],
                "predicted_class": class_name,
                "confidence": float(prediction['confidence']),
                "image_path": prediction['filename'].split(BASE_DIR.split('/app')[0])[-1],
                "ground_truth": prediction.get('ground_truth', 'unknown'),
                "original_index": prediction['original_index']
            }
            formatted_predictions.append(formatted_prediction)

        # Save predictions to file for later retrieval
        predictions_file = os.path.join(video_prediction_dir, "predictions.json")
        with open(predictions_file, 'w') as f:
            json.dump(formatted_predictions, f, indent=2)
        print(formatted_predictions)
        return PredictionResponse(
            status="success",
            video_name=video_name,
            total_predictions=len(formatted_predictions),
            predictions_by_class=predictions_by_class,
            predictions=formatted_predictions
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting frames: {str(e)}")


@app.get("/predictions/{video_name}")
async def get_video_predictions(
        video_name: str,
        include_normal: bool = Query(True, description="Include normal predictions"),
        class_filter: Optional[str] = Query(None, description="Filter by specific class"),
        confidence_threshold: Optional[float] = Query(None, description="Minimum confidence threshold")
):
    """
    Retrieve existing predictions for a specific video.

    Args:
        video_name: Name of the video
        include_normal: Whether to include normal predictions (default: False)
        class_filter: Filter results by specific class
        confidence_threshold: Minimum confidence score to include (0.0-1.0)

    Returns:
        JSON with existing predictions for the video
    """
    try:
        # Load predictions from file
        predictions_file = os.path.join(PREDICTIONS_BASE_DIR, video_name, "predictions.json")

        if not os.path.exists(predictions_file):
            raise HTTPException(
                status_code=404,
                detail=f"No predictions found for video '{video_name}'. Please run predictions first."
            )

        with open(predictions_file, 'r') as f:
            predictions = json.load(f)

        # Apply filters
        filtered_predictions = predictions.copy()

        # Filter out normal predictions if not requested
        if not include_normal:
            filtered_predictions = [p for p in filtered_predictions if p['predicted_class'].lower() != 'normal']

        # Apply class filter
        if class_filter:
            filtered_predictions = [p for p in filtered_predictions if
                                    p['predicted_class'].lower() == class_filter.lower()]

        # Apply confidence threshold
        if confidence_threshold is not None:
            filtered_predictions = [p for p in filtered_predictions if p['confidence'] >= confidence_threshold]
        print(filtered_predictions)
        return {
            "video_name": video_name,
            "total_predictions": len(predictions),
            "filtered_predictions": len(filtered_predictions),
            "filters_applied": {
                "include_normal": include_normal,
                "class_filter": class_filter,
                "confidence_threshold": confidence_threshold
            },
            "predictions": filtered_predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving predictions: {str(e)}")


@app.post("/feedback/expert", response_model=FeedbackResponse)
async def submit_expert_feedback(
        frame_image: UploadFile = File(...),
        frame_id: str = Form(...),
        video_name: str = Form(...),
        model_prediction: str = Form(...),
        expert_classification: str = Form(...),
        confidence_score: Optional[float] = Form(None),
        expert_notes: Optional[str] = Form(None)
):
    """
    Submit expert feedback for active learning.

    Args:
        frame_image: The frame image file
        frame_id: Unique frame identifier
        video_name: Name of the source video
        model_prediction: Model's original prediction
        expert_classification: Expert's classification
        confidence_score: Expert's confidence in their classification
        expert_notes: Additional notes from expert

    Returns:
        FeedbackResponse with feedback details and retraining status
    """
    try:
        # Validate expert classification
        valid_classifications = ["normal", "crack", "corrosion", "leakage"]
        if expert_classification.lower() not in valid_classifications:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid classification. Must be one of: {valid_classifications}"
            )

        # Generate unique feedback ID
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Save image to appropriate class folder
        classification_folder = Path(FEEDBACK_DATA_DIR) / expert_classification.lower()
        image_filename = f"{feedback_id}_{frame_id}.jpg"
        image_path = classification_folder / image_filename

        # Save the uploaded image
        contents = await frame_image.read()
        with open(image_path, "wb") as f:
            f.write(contents)

        # Create annotation entry
        annotation = {
            "feedback_id": feedback_id,
            "frame_id": frame_id,
            "video_name": video_name,
            "image_path": str(image_path),
            "model_prediction": model_prediction,
            "expert_classification": expert_classification.lower(),
            "confidence_score": confidence_score,
            "expert_notes": expert_notes,
            "timestamp": timestamp,
            "is_correction": model_prediction.lower() != expert_classification.lower()
        }

        # Load existing annotations and add new one
        annotations = load_annotations()
        annotations.append(annotation)
        save_annotations(annotations)

        # Save to existing feedback system for compatibility
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
            message=f"Expert feedback recorded successfully. Image saved to {expert_classification} folder.",
            feedback_id=feedback_id,
            samples_collected=samples_collected,
            ready_for_retrain=ready_for_retrain
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing expert feedback: {str(e)}")


@app.post("/retrain")
async def retrain_model():
    """
    Retrain the CBM model using accumulated expert feedback.

    Returns:
        Status of the retraining process
    """
    try:
        annotations = load_annotations()

        if len(annotations) < RETRAIN_TRIGGER_COUNT:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough samples for retraining. Need {RETRAIN_TRIGGER_COUNT}, have {len(annotations)}"
            )

        # Use active learning data for retraining
        print(f"Retraining with {len(annotations)} active learning samples")
        retrain_cbm(FEEDBACK_DATA_DIR, "static/images", "feedback")

        # Reload the model
        global model
        model = load_cbm(TRAINED_MODEL_DIR, 'cpu')

        return {
            "status": "success",
            "message": "Model retrained successfully",
            "active_learning_samples": len(annotations),
            "model_updated": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining model: {str(e)}")


@app.get("/feedback/stats")
async def get_feedback_stats():
    """Get statistics about collected expert feedback"""
    try:
        annotations = load_annotations()

        if not annotations:
            return {
                "total_samples": 0,
                "by_classification": {},
                "corrections_count": 0,
                "ready_for_retrain": False,
                "correction_rate":0,
                "retrain_threshold":0

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


@app.get("/videos")
async def list_processed_videos():
    """List all processed videos and their status"""
    try:
        videos = []

        # Check frames directory
        if os.path.exists(FRAMES_BASE_DIR):
            for video_dir in os.listdir(FRAMES_BASE_DIR):
                video_path = os.path.join(FRAMES_BASE_DIR, video_dir)
                if os.path.isdir(video_path):
                    frame_count = len([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])

                    # Check if predictions exist
                    predictions_path = os.path.join(PREDICTIONS_BASE_DIR, video_dir, "predictions.json")
                    has_predictions = os.path.exists(predictions_path)

                    prediction_count = 0
                    if has_predictions:
                        with open(predictions_path, 'r') as f:
                            predictions = json.load(f)
                            prediction_count = len(predictions)

                    videos.append({
                        "video_name": video_dir,
                        "frame_count": frame_count,
                        "has_predictions": has_predictions,
                        "prediction_count": prediction_count
                    })

        return {
            "total_videos": len(videos),
            "videos": videos
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing videos: {str(e)}")


# Initialize directories on startup
@app.on_event("startup")
async def startup_event():
    ensure_directories()
    print(f"CBM Video Processing API initialized")
    print(f"Video upload directory: {VIDEO_UPLOAD_DIR}")
    print(f"Frames base directory: {FRAMES_BASE_DIR}")
    print(f"Predictions base directory: {PREDICTIONS_BASE_DIR}")
    print(f"Feedback data directory: {FEEDBACK_DATA_DIR}")
    print(f"Trained model directory: {TRAINED_MODEL_DIR}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)