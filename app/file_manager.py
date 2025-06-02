import os
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging
import cv2

class CBMFileManager:
    """
    Centralized file management for CBM workflow
    """

    def __init__(self, base_dir: str = "shared_data"):
        self.base_dir = Path(base_dir)
        self.setup_directory_structure()
        self.setup_logging()

    def setup_directory_structure(self):
        """Create the recommended directory structure"""
        self.dirs = {
            'incoming': self.base_dir / "1_incoming",
            'extracted_frames': self.base_dir / "2_extracted_frames",
            'predictions': self.base_dir / "3_predictions",
            'expert_feedback': self.base_dir / "4_expert_feedback",
            'training_data': self.base_dir / "5_training_data",
            'models': self.base_dir / "6_models",
            'reports': self.base_dir / "7_reports"
        }

        # Subdirectories
        self.subdirs = {
            'pending_review': self.dirs['expert_feedback'] / "pending_review",
            'expert_corrections': self.dirs['expert_feedback'] / "expert_corrections",
            'training_current': self.dirs['training_data'] / "current",
            'training_staging': self.dirs['training_data'] / "staging",
            'training_versions': self.dirs['training_data'] / "versions",
            'model_current': self.dirs['models'] / "current",
            'model_staging': self.dirs['models'] / "staging",
            'model_archive': self.dirs['models'] / "archive"
        }

        # Create all directories
        for dir_path in list(self.dirs.values()) + list(self.subdirs.values()):
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create class subdirectories
        self.classes = ['normal', 'crack', 'corrosion', 'leakage']
        for class_name in self.classes:
            for base_dir in [self.subdirs['expert_corrections'],
                             self.subdirs['training_current'],
                             self.subdirs['training_staging']]:
                (base_dir / class_name).mkdir(exist_ok=True)

    def setup_logging(self):
        """Setup logging for file operations"""
        log_file = self.dirs['reports'] / "file_operations.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)


# ==========================================
# STAGE 1: VIDEO UPLOAD AND FRAME EXTRACTION
# ==========================================

class VideoProcessor(CBMFileManager):
    """Handle video upload and frame extraction"""

    def process_uploaded_video(self, video_file, video_name: str, frame_interval: int = 30):
        """
        Process uploaded video through the complete pipeline

        Returns:
            dict: Processing results with file paths and metadata
        """
        try:
            # Step 1: Save uploaded video
            video_path = self.save_uploaded_video(video_file, video_name)

            # Step 2: Extract frames
            frames_info = self.extract_frames(video_path, video_name, frame_interval)

            # Step 3: Create metadata
            metadata = self.create_video_metadata(video_name, video_path, frames_info, frame_interval)

            self.logger.info(f"Successfully processed video: {video_name}")
            return {
                'video_path': str(video_path),
                'frames_dir': str(self.dirs['extracted_frames'] / video_name),
                'frame_count': len(frames_info),
                'metadata': metadata
            }

        except Exception as e:
            self.logger.error(f"Error processing video {video_name}: {e}")
            raise

    def save_uploaded_video(self, video_file, video_name: str) -> Path:
        """Save uploaded video to incoming directory"""
        video_path = self.dirs['incoming'] / f"{video_name}.mp4"

        with open(video_path, 'wb') as f:
            shutil.copyfileobj(video_file.file, f)

        self.logger.info(f"Saved video: {video_path}")
        return video_path

    def extract_frames(self, video_path: Path, video_name: str, frame_interval: int) -> List[Dict]:
        """Extract frames and save to extracted_frames directory"""
        frames_dir = self.dirs['extracted_frames'] / video_name
        frames_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(frame_count)
        success, frame_number = True, 0
        # Your existing frame extraction logic here
        # This is a placeholder - use your actual video processing code
        extracted_frames = []

        # Example: extract frames using your existing logic
        # frames = extract_frames_from_video(video_path, frame_interval)
        while success:
            success, frame = cap.read()
            # print("Reached here " + str(video_path))
            # print(frame)
            if frame_number % frame_interval == 0 and success:
                frame_filename = f"frame_{frame_number:06d}.jpg"
                frame_path = frames_dir / frame_filename
                # frame_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")
                cv2.imwrite(str(frame_path), frame)
                frame_info = {
                    'frame_id': f"{video_name}_frame_{frame_number:06d}",
                    'filename': frame_filename,
                    'path': str(frame_path),
                    'timestamp': frame_number,
                    'video_name': video_name
                }
                extracted_frames.append(frame_info)
            frame_number += 1

        cap.release()
        # Save each frame with consistent naming
        # for i, frame in enumerate(extracted_frames):
        #     frame_filename = f"frame_{i:06d}.jpg"
        #     frame_path = frames_dir / frame_filename
        #
        #     # Save frame
        #     # frame.save(frame_path)
        #
        #     frame_info = {
        #         'frame_id': f"{video_name}_frame_{i:06d}",
        #         'filename': frame_filename,
        #         'path': str(frame_path),
        #         'timestamp': i * frame_interval,
        #         'video_name': video_name
        #     }
        #     extracted_frames.append(frame_info)

        return extracted_frames

    def create_video_metadata(self, video_name: str, video_path: Path,
                              frames_info: List[Dict], frame_interval: int) -> Dict:
        """Create and save metadata for the video processing"""
        metadata = {
            'video_name': video_name,
            'video_path': str(video_path),
            'processed_at': datetime.now().isoformat(),
            'frame_interval': frame_interval,
            'total_frames': len(frames_info),
            'frames': frames_info
        }

        # Save metadata
        metadata_path = self.dirs['extracted_frames'] / video_name / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata


# ==========================================
# STAGE 2: MODEL PREDICTIONS
# ==========================================

class PredictionManager(CBMFileManager):
    """Handle model predictions and organization"""

    def process_video_predictions(self, video_name: str, model_predictions: List[Dict]) -> Dict:
        """
        Process and organize model predictions

        Args:
            video_name: Name of the video
            model_predictions: List of prediction dictionaries from your model

        Returns:
            dict: Organized prediction results
        """
        try:
            # Create prediction directory for this video
            pred_dir = self.dirs['predictions'] / video_name
            pred_dir.mkdir(exist_ok=True)

            # Create class subdirectories
            for class_name in self.classes:
                (pred_dir / class_name).mkdir(exist_ok=True)

            # Process each prediction
            organized_predictions = []
            for pred in model_predictions:
                organized_pred = self.organize_prediction(pred, video_name, pred_dir)
                organized_predictions.append(organized_pred)

            # Save predictions metadata
            predictions_metadata = {
                'video_name': video_name,
                'predicted_at': datetime.now().isoformat(),
                'total_predictions': len(organized_predictions),
                'class_distribution': self.get_class_distribution(organized_predictions),
                'predictions': organized_predictions
            }

            # Save to predictions.json
            pred_file = pred_dir / "predictions.json"
            with open(pred_file, 'w') as f:
                json.dump(predictions_metadata, f, indent=2)

            # Identify uncertain predictions for expert review
            self.flag_uncertain_predictions(organized_predictions, video_name)

            self.logger.info(f"Processed {len(organized_predictions)} predictions for {video_name}")
            return predictions_metadata

        except Exception as e:
            self.logger.error(f"Error processing predictions for {video_name}: {e}")
            raise

    def organize_prediction(self, prediction: Dict, video_name: str, pred_dir: Path) -> Dict:
        """Organize a single prediction"""
        # Get original frame path
        frame_path = Path(prediction['filename'])
        predicted_class = prediction['class']
        confidence = float(prediction['confidence'])

        # Create organized prediction entry
        organized_pred = {
            'frame_id': prediction.get('frame_id', f"{video_name}_frame_{uuid.uuid4().hex[:8]}"),
            'original_frame_path': str(frame_path),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'video_name': video_name,
            'prediction_timestamp': datetime.now().isoformat(),
            'needs_review': confidence < 0.7,  # Flag low confidence for review
            'features': prediction.get('features', []),
            'values': prediction.get('values', [])
        }

        # Copy/link frame to predicted class folder
        if frame_path.exists():
            dest_path = pred_dir / predicted_class / frame_path.name

            # Use hard link to save space (or copy if preferred)
            try:
                dest_path.hardlink_to(frame_path)
                organized_pred['organized_path'] = str(dest_path)
            except:
                shutil.copy2(frame_path, dest_path)
                organized_pred['organized_path'] = str(dest_path)

        return organized_pred

    def flag_uncertain_predictions(self, predictions: List[Dict], video_name: str):
        """Flag uncertain predictions for expert review"""
        uncertain_threshold = 0.7
        uncertain_predictions = [
            p for p in predictions
            if p['confidence'] < uncertain_threshold or p['predicted_class'] != 'normal'
        ]

        if uncertain_predictions:
            review_queue_file = self.subdirs['pending_review'] / "review_queue.json"

            # Load existing queue or create new
            if review_queue_file.exists():
                with open(review_queue_file, 'r') as f:
                    review_queue = json.load(f)
            else:
                review_queue = {'pending_items': [], 'created_at': datetime.now().isoformat()}

            # Add uncertain predictions to queue
            for pred in uncertain_predictions:
                review_item = {
                    'frame_id': pred['frame_id'],
                    'video_name': video_name,
                    'predicted_class': pred['predicted_class'],
                    'confidence': pred['confidence'],
                    'reason': 'low_confidence' if pred['confidence'] < uncertain_threshold else 'potential_defect',
                    'added_at': datetime.now().isoformat(),
                    'status': 'pending'
                }
                review_queue['pending_items'].append(review_item)

                # Copy frame to pending review folder
                if Path(pred['organized_path']).exists():
                    review_frame_path = self.subdirs['pending_review'] / f"{pred['frame_id']}.jpg"
                    shutil.copy2(pred['organized_path'], review_frame_path)

            # Save updated queue
            with open(review_queue_file, 'w') as f:
                json.dump(review_queue, f, indent=2)

            self.logger.info(f"Flagged {len(uncertain_predictions)} frames for expert review")

    def get_class_distribution(self, predictions: List[Dict]) -> Dict[str, int]:
        """Get distribution of predictions by class"""
        distribution = {}
        for pred in predictions:
            class_name = pred['predicted_class']
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


# ==========================================
# STAGE 3: EXPERT FEEDBACK AND ACTIVE LEARNING
# ==========================================

class ExpertFeedbackManager(CBMFileManager):
    """Handle expert feedback and active learning"""

    def submit_expert_feedback(self, frame_id: str, expert_classification: str,
                               model_prediction: str, confidence: float,
                               expert_notes: str = "", frame_image_data: bytes = None) -> Dict:
        """
        Process expert feedback and update training data

        Returns:
            dict: Feedback processing results
        """
        try:
            # Create feedback entry
            feedback_entry = {
                'feedback_id': str(uuid.uuid4()),
                'frame_id': frame_id,
                'expert_classification': expert_classification,
                'model_prediction': model_prediction,
                'expert_confidence': confidence,
                'expert_notes': expert_notes,
                'is_correction': expert_classification != model_prediction,
                'submitted_at': datetime.now().isoformat(),
                'processed': False
            }

            # Save frame to expert corrections folder
            if frame_image_data:
                frame_path = self.save_expert_corrected_frame(
                    frame_image_data, frame_id, expert_classification
                )
                feedback_entry['corrected_frame_path'] = str(frame_path)

            # Log feedback
            self.log_expert_feedback(feedback_entry)

            # Update training data staging
            self.update_training_data_staging(feedback_entry)

            # Remove from pending review if it was there
            self.remove_from_pending_review(frame_id)

            # Check if ready for retraining
            ready_for_retrain = self.check_retraining_readiness()

            self.logger.info(f"Processed expert feedback for {frame_id}")

            return {
                'feedback_id': feedback_entry['feedback_id'],
                'is_correction': feedback_entry['is_correction'],
                'ready_for_retrain': ready_for_retrain,
                'training_samples_count': self.get_training_samples_count()
            }

        except Exception as e:
            self.logger.error(f"Error processing expert feedback for {frame_id}: {e}")
            raise

    def save_expert_corrected_frame(self, frame_data: bytes, frame_id: str,
                                    expert_class: str) -> Path:
        """Save expert-corrected frame to appropriate class folder"""
        class_dir = self.subdirs['expert_corrections'] / expert_class
        frame_path = class_dir / f"{frame_id}.jpg"

        with open(frame_path, 'wb') as f:
            f.write(frame_data)

        return frame_path

    def log_expert_feedback(self, feedback_entry: Dict):
        """Log expert feedback to audit trail"""
        log_file = self.dirs['expert_feedback'] / "feedback_log.json"

        if log_file.exists():
            with open(log_file, 'r') as f:
                feedback_log = json.load(f)
        else:
            feedback_log = {'entries': [], 'created_at': datetime.now().isoformat()}

        feedback_log['entries'].append(feedback_entry)
        feedback_log['last_updated'] = datetime.now().isoformat()

        with open(log_file, 'w') as f:
            json.dump(feedback_log, f, indent=2)

    def update_training_data_staging(self, feedback_entry: Dict):
        """Update staging training data with expert feedback"""
        if 'corrected_frame_path' not in feedback_entry:
            return

        source_path = Path(feedback_entry['corrected_frame_path'])
        expert_class = feedback_entry['expert_classification']

        # Copy to staging directory
        staging_class_dir = self.subdirs['training_staging'] / expert_class
        dest_path = staging_class_dir / source_path.name

        shutil.copy2(source_path, dest_path)

        # Create training metadata
        training_metadata = {
            'frame_id': feedback_entry['frame_id'],
            'expert_classification': expert_class,
            'expert_confidence': feedback_entry['expert_confidence'],
            'model_prediction': feedback_entry['model_prediction'],
            'is_correction': feedback_entry['is_correction'],
            'expert_notes': feedback_entry['expert_notes'],
            'added_to_training': datetime.now().isoformat()
        }

        # Save metadata alongside frame
        metadata_path = staging_class_dir / f"{source_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)

    def check_retraining_readiness(self, min_samples: int = 50) -> bool:
        """Check if enough samples collected for retraining"""
        total_samples = 0

        for class_name in self.classes:
            class_dir = self.subdirs['training_staging'] / class_name
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            total_samples += len(image_files)

        return total_samples >= min_samples

    def get_training_samples_count(self) -> Dict[str, int]:
        """Get count of training samples by class"""
        counts = {}
        total = 0

        for class_name in self.classes:
            class_dir = self.subdirs['training_staging'] / class_name
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            counts[class_name] = len(image_files)
            total += len(image_files)

        counts['total'] = total
        return counts


# ==========================================
# STAGE 4: MODEL RETRAINING
# ==========================================

class ModelRetrainingManager(CBMFileManager):
    """Handle model retraining workflow"""

    def prepare_retraining_data(self) -> Dict:
        """Prepare data for model retraining"""
        try:
            # Move staging data to current training data
            training_stats = self.consolidate_training_data()

            # Create training dataset version
            version = self.create_training_version()

            # Prepare training configuration
            training_config = self.create_training_config(training_stats, version)

            self.logger.info(f"Prepared retraining data: version {version}")

            return {
                'version': version,
                'training_stats': training_stats,
                'config': training_config,
                'ready': True
            }

        except Exception as e:
            self.logger.error(f"Error preparing retraining data: {e}")
            raise

    def consolidate_training_data(self) -> Dict:
        """Move staging data to current training directory"""
        stats = {'moved_files': 0, 'class_counts': {}}

        for class_name in self.classes:
            staging_dir = self.subdirs['training_staging'] / class_name
            current_dir = self.subdirs['training_current'] / class_name

            # Move files from staging to current
            moved_count = 0
            for file_path in staging_dir.glob("*.jpg"):
                dest_path = current_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
                moved_count += 1

            # Move metadata files too
            for metadata_path in staging_dir.glob("*_metadata.json"):
                dest_path = current_dir / metadata_path.name
                shutil.move(str(metadata_path), str(dest_path))

            stats['class_counts'][class_name] = moved_count
            stats['moved_files'] += moved_count

        return stats

    def create_training_version(self) -> str:
        """Create a versioned snapshot of training data"""
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_dir = self.subdirs['training_versions'] / version

        # Copy current training data to version directory
        shutil.copytree(self.subdirs['training_current'], version_dir)

        return version

    def archive_current_model(self) -> str:
        """Archive current model before retraining"""
        if not (self.subdirs['model_current'] / "model.pt").exists():
            return None

        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        archive_dir = self.subdirs['model_archive'] / version

        # Copy current model to archive
        shutil.copytree(self.subdirs['model_current'], archive_dir)

        self.logger.info(f"Archived current model as {version}")
        return version


# ==========================================
# USAGE EXAMPLE AND INTEGRATION
# ==========================================

class CBMWorkflowManager:
    """
    Complete workflow manager integrating all stages
    """

    def __init__(self, base_dir: str = "shared_data"):
        self.video_processor = VideoProcessor(base_dir)
        self.prediction_manager = PredictionManager(base_dir)
        self.feedback_manager = ExpertFeedbackManager(base_dir)
        self.retraining_manager = ModelRetrainingManager(base_dir)

    def process_complete_workflow(self, video_file, video_name: str,
                                  frame_interval: int = 30) -> Dict:
        """
        Execute complete workflow from video upload to ready for feedback
        """
        results = {}

        try:
            # Stage 1: Process video and extract frames
            video_results = self.video_processor.process_uploaded_video(
                video_file, video_name, frame_interval
            )
            results['video_processing'] = video_results
            print(results)
            # Stage 2: Run model predictions (you'll integrate your model here)
            # model_predictions = your_model.predict(video_results['frames_dir'])
            # prediction_results = self.prediction_manager.process_video_predictions(
            #     video_name, model_predictions
            # )
            # results['predictions'] = prediction_results

            return results

        except Exception as e:
            results['error'] = str(e)
            return results
