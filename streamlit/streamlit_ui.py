import streamlit as st
import requests
import json
from PIL import Image
import io
import pandas as pd
from datetime import datetime
import time
import os
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import base64
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
VALID_VIDEO_FORMATS = ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv']
VALID_IMAGE_FORMATS = ['png', 'jpg', 'jpeg']
CLASSIFICATION_CLASSES = ["normal", "crack", "corrosion", "leakage"]

# Page configuration
st.set_page_config(
    page_title="CBM Expert Feedback System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .status-good {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


class APIClient:
    """Centralized API client for all backend communications"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> Dict:
        """Check API health and connectivity"""
        try:
            response = self.session.get(f"{self.base_url}/feedback/stats", timeout=5)
            return {
                "status": "connected" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            return {"status": "disconnected", "error": str(e)}

    def upload_video(self, file_data: bytes, filename: str, frame_interval: int = 50) -> Dict:
        """Upload video and extract frames"""
        try:
            files = {'file': (filename, file_data, 'video/mp4')}
            data = {'frame_interval': frame_interval}

            response = self.session.post(
                f"{self.base_url}/upload_video",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout for large videos
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.text}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict_frames(self, video_name: str) -> Dict:
        """Run predictions on video frames"""
        try:
            response = self.session.post(
                f"{self.base_url}/predict_frames/{video_name}",
                timeout=300
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.text}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_predictions(self, video_name: str, include_normal: bool = False,
                        class_filter: str = None, confidence_threshold: float = None) -> Dict:
        """Get existing predictions for a video"""
        try:
            params = {"include_normal": include_normal}
            if class_filter:
                params["class_filter"] = class_filter
            if confidence_threshold:
                params["confidence_threshold"] = confidence_threshold

            response = self.session.get(
                f"{self.base_url}/predictions/{video_name}",
                params=params
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.text}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def submit_feedback(self, image_data: bytes, frame_id: str, video_name: str,
                        model_prediction: str, expert_classification: str,
                        confidence_score: float = None, expert_notes: str = None) -> Dict:
        """Submit expert feedback"""
        try:
            files = {'frame_image': ('frame.jpg', image_data, 'image/jpeg')}
            data = {
                'frame_id': frame_id,
                'video_name': video_name,
                'model_prediction': model_prediction,
                'expert_classification': expert_classification,
                'confidence_score': confidence_score,
                'expert_notes': expert_notes or ""
            }

            response = self.session.post(
                f"{self.base_url}/feedback/expert",
                files=files,
                data=data
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.text}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics"""
        try:
            response = self.session.get(f"{self.base_url}/feedback/stats")

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.text}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def retrain_model(self) -> Dict:
        """Trigger model retraining"""
        try:
            response = self.session.post(f"{self.base_url}/retrain", timeout=1800)  # 30 min timeout

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.text}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_videos(self) -> Dict:
        """List all processed videos"""
        try:
            response = self.session.get(f"{self.base_url}/videos")

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.text}

        except Exception as e:
            return {"success": False, "error": str(e)}


class SessionManager:
    """Manage Streamlit session state"""

    @staticmethod
    def init_session():
        """Initialize session state variables"""
        defaults = {
            'api_client': APIClient(API_BASE_URL),
            'current_video': None,
            'video_predictions': None,
            'feedback_stats': None,
            'processed_videos': [],
            'selected_video_for_feedback': None,
            'feedback_filters': {
                'include_normal': False,
                'class_filter': None,
                'confidence_threshold': None
            }
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def clear_video_data():
        """Clear video-related session data"""
        keys_to_clear = ['current_video', 'video_predictions']
        for key in keys_to_clear:
            if key in st.session_state:
                st.session_state[key] = None

    @staticmethod
    def update_feedback_stats():
        """Update feedback statistics in session"""
        result = st.session_state.api_client.get_feedback_stats()
        if result["success"]:
            st.session_state.feedback_stats = result["data"]
        return result


class UIComponents:
    """Reusable UI components"""

    @staticmethod
    def render_status_indicator():
        """Render API connection status in sidebar"""
        with st.sidebar:
            st.markdown("### 🔌 System Status")

            health = st.session_state.api_client.health_check()

            if health["status"] == "connected":
                st.markdown('<div class="status-good">✅ API Connected</div>', unsafe_allow_html=True)
                if health.get("data"):
                    stats = health["data"]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Samples", stats.get('total_samples', 0))
                    with col2:
                        correction_rate = stats.get('correction_rate', 0) * 100
                        st.metric("Corrections", f"{correction_rate:.1f}%")
            elif health["status"] == "error":
                st.markdown('<div class="status-warning">⚠️ API Error</div>', unsafe_allow_html=True)
                st.caption(f"Status Code: {health.get('status_code', 'Unknown')}")
            else:
                st.markdown('<div class="status-error">❌ API Disconnected</div>', unsafe_allow_html=True)
                st.caption(f"Error: {health.get('error', 'Unknown')}")

    @staticmethod
    def render_video_selector():
        """Render video selection dropdown"""
        result = st.session_state.api_client.list_videos()

        if result["success"] and result["data"]["videos"]:
            videos = result["data"]["videos"]

            options = ["Select a video..."] + [v["video_name"] for v in videos]

            selected = st.selectbox(
                "Choose a processed video:",
                options,
                key="video_selector"
            )

            if selected != "Select a video...":
                # Find selected video data
                video_data = next((v for v in videos if v["video_name"] == selected), None)
                return selected, video_data

        return None, None

    @staticmethod
    def render_prediction_filters():
        """Render prediction filter controls"""
        col1, col2, col3 = st.columns(3)

        with col1:
            include_normal = st.checkbox(
                "Include Normal Predictions",
                value=st.session_state.feedback_filters['include_normal'],
                key="filter_include_normal"
            )

        with col2:
            class_filter = st.selectbox(
                "Filter by Class:",
                ["All"] + CLASSIFICATION_CLASSES,
                index=0,
                key="filter_class"
            )

        with col3:
            confidence_threshold = st.slider(
                "Min Confidence:",
                0.0, 1.0, 0.5,
                step=0.1,
                key="filter_confidence"
            )

        # Update session state
        st.session_state.feedback_filters.update({
            'include_normal': include_normal,
            'class_filter': class_filter if class_filter != "All" else None,
            'confidence_threshold': confidence_threshold
        })

        return include_normal, class_filter, confidence_threshold

    @staticmethod
    def render_prediction_card(prediction: Dict, index: int):
        """Render a single prediction card with feedback form"""
        confidence = float(prediction['confidence'])
        pred_class = prediction['predicted_class']

        # Determine card styling based on prediction
        if pred_class.lower() == 'normal':
            card_color = "🟢"
            priority = "low"
        elif confidence > 0.8:
            card_color = "🔴"
            priority = "high"
        elif confidence > 0.6:
            card_color = "🟡"
            priority = "medium"
        else:
            card_color = "🟠"
            priority = "review"

        # Expand high-priority items by default
        expanded = priority in ["high", "review"]

        with st.expander(
                f"{card_color} Frame {index + 1}: {pred_class.title()} ({confidence:.1%})",
                expanded=expanded
        ):
            col1, col2 = st.columns([1, 2])

            with col1:
                # Display frame image
                try:
                    image_url = f"{API_BASE_URL}{prediction['image_path']}"
                    st.image(image_url, caption=f"Frame {index + 1}", use_container_width=True)

                    # Display prediction metrics
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.metric("Predicted Class", pred_class.title())

                    if prediction.get('ground_truth') and prediction['ground_truth'] != 'unknown':
                        st.metric("Ground Truth", prediction['ground_truth'].title())

                except Exception:
                    st.error("Could not load image")
                    st.code(prediction['image_path'])

            with col2:
                # Feedback form
                with st.form(f"feedback_form_{prediction['frame_id']}"):
                    st.markdown("**Expert Review:**")

                    # Expert classification
                    expert_class = st.selectbox(
                        "Correct Classification:",
                        CLASSIFICATION_CLASSES,
                        index=CLASSIFICATION_CLASSES.index(
                            pred_class.lower()) if pred_class.lower() in CLASSIFICATION_CLASSES else 0,
                        key=f"expert_class_{prediction['frame_id']}"
                    )

                    # Expert confidence and priority
                    col_a, col_b = st.columns(2)
                    with col_a:
                        expert_confidence = st.slider(
                            "Your Confidence:",
                            0.0, 1.0, 0.9,
                            step=0.1,
                            key=f"expert_conf_{prediction['frame_id']}"
                        )

                    with col_b:
                        priority_level = st.selectbox(
                            "Priority:",
                            ["Low", "Medium", "High", "Critical"],
                            index=1,
                            key=f"priority_{prediction['frame_id']}"
                        )

                    # Expert notes
                    expert_notes = st.text_area(
                        "Notes:",
                        placeholder="Describe the defect, location, severity, recommended action...",
                        height=100,
                        key=f"expert_notes_{prediction['frame_id']}"
                    )

                    # Additional metadata
                    # with st.expander("Additional Details (Optional)"):
                    #     location = st.text_input(
                    #         "Location/Asset ID:",
                    #         key=f"location_{prediction['frame_id']}"
                    #     )
                    #
                    #     severity = st.selectbox(
                    #         "Severity Level:",
                    #         ["Low", "Medium", "High", "Critical"],
                    #         key=f"severity_{prediction['frame_id']}"
                    #     )
                    #
                    #     action_required = st.checkbox(
                    #         "Immediate Action Required",
                    #         key=f"action_{prediction['frame_id']}"
                    #     )

                    # Submit button
                    col_submit, col_info = st.columns([1, 1])
                    with col_submit:
                        submitted = st.form_submit_button(
                            "Submit Feedback",
                            type="primary" if expert_class != pred_class else "secondary"
                        )

                    with col_info:
                        if expert_class != pred_class:
                            st.info("🔄 Correction")
                        else:
                            st.success("✅ Confirmation")

                    # Handle form submission
                    if submitted:
                        # Compile comprehensive notes
                        comprehensive_notes = expert_notes
                        # if location:
                        #     comprehensive_notes += f"\nLocation: {location}"
                        # if priority_level != "Medium":
                        #     comprehensive_notes += f"\nPriority: {priority_level}"
                        # if severity != "Medium":
                        #     comprehensive_notes += f"\nSeverity: {severity}"
                        # if action_required:
                        #     comprehensive_notes += "\nIMMEDIATE ACTION REQUIRED"

                        UIComponents.submit_feedback_for_prediction(
                            prediction, expert_class, expert_confidence, comprehensive_notes
                        )

    @staticmethod
    def submit_feedback_for_prediction(prediction: Dict, expert_class: str,
                                       confidence: float, notes: str):
        """Submit feedback for a prediction"""
        try:
            # Download the image
            image_url = f"{API_BASE_URL}{prediction['image_path']}"
            image_response = requests.get(image_url)

            if image_response.status_code != 200:
                st.error("Could not download image for feedback")
                return

            # Submit feedback
            result = st.session_state.api_client.submit_feedback(
                image_response.content,
                prediction['frame_id'],
                prediction.get('video_name', ''),
                prediction['predicted_class'],
                expert_class,
                confidence,
                notes
            )

            if result["success"]:
                feedback_data = result["data"]

                # Show success message
                if expert_class != prediction['predicted_class']:
                    st.success(f"✅ Correction submitted: {prediction['predicted_class']} → {expert_class}")
                else:
                    st.success(f"✅ Confirmation submitted: {expert_class}")

                # Show feedback summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Feedback ID", feedback_data['feedback_id'][:8] + "...")
                with col2:
                    st.metric("Total Samples", feedback_data['samples_collected'])
                with col3:
                    st.metric("Ready for Retrain", "Yes" if feedback_data['ready_for_retrain'] else "No")

                if feedback_data['ready_for_retrain']:
                    st.info("🔄 System ready for retraining!")

                # Update stats in session
                SessionManager.update_feedback_stats()

                # Show progress
                st.balloons()

            else:
                st.error(f"Error submitting feedback: {result['error']}")

        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")


def render_dashboard():
    """Render main dashboard with system overview"""
    st.markdown('<h1 class="main-header">🔧 CBM Expert Feedback System</h1>', unsafe_allow_html=True)
    st.markdown("Condition-Based Monitoring with AI-powered defect detection and expert feedback")

    # Update and display key metrics
    SessionManager.update_feedback_stats()

    if st.session_state.feedback_stats:
        stats = st.session_state.feedback_stats

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Samples",
                stats['total_samples'],
                delta=None
            )

        with col2:
            st.metric(
                "Corrections",
                stats['corrections_count'],
                delta=f"{stats['correction_rate']:.1%} rate"
            )

        with col3:
            ready_status = "Ready" if stats['ready_for_retrain'] else "Collecting"
            st.metric(
                "Retraining Status",
                ready_status,
                delta=f"{stats['total_samples']}/{stats['retrain_threshold']}"
            )

        with col4:
            if stats['by_classification']:
                defect_count = sum(count for class_name, count in stats['by_classification'].items()
                                   if class_name != 'normal')
                st.metric(
                    "Defect Samples",
                    defect_count,
                    delta=f"{(defect_count / stats['total_samples'] * 100):.1f}%" if stats[
                                                                                         'total_samples'] > 0 else None
                )

        # Progress toward retraining
        st.markdown("### Retraining Progress")
        if stats['retrain_threshold']==0:
            progress = 0
        else:
            progress = min(stats['total_samples'] / stats['retrain_threshold'], 1.0)
        st.progress(progress)

        remaining = max(0, stats['retrain_threshold'] - stats['total_samples'])
        if remaining > 0:
            st.info(f"📊 {remaining} more samples needed for retraining")
        else:
            st.success("🎯 Ready for model retraining!")

        # Class distribution visualization
        if stats['by_classification']:
            st.markdown("### Sample Distribution")

            # Create pie chart
            fig = px.pie(
                values=list(stats['by_classification'].values()),
                names=list(stats['by_classification'].keys()),
                title="Feedback Samples by Classification"
            )
            st.plotly_chart(fig, use_container_width=True)


def render_video_processing():
    """Render video processing interface"""
    st.markdown('<h2 class="section-header">🎥 Video Processing & Analysis</h2>', unsafe_allow_html=True)

    # Create tabs for different video processing functions
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📋 Manage Videos", "🔍 Advanced Analysis"])

    with tab1:
        render_video_upload()

    with tab2:
        render_video_management()

    with tab3:
        render_advanced_analysis()


def render_video_upload():
    """Render video upload interface"""
    st.markdown("### Upload New Video")

    # Upload section
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=VALID_VIDEO_FORMATS,
        help=f"Supported formats: {', '.join(VALID_VIDEO_FORMATS).upper()}"
    )

    if uploaded_video:
        # Display video information
        file_size_mb = uploaded_video.size / (1024 * 1024)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Filename:** {uploaded_video.name}")
        with col2:
            st.info(f"**Size:** {file_size_mb:.2f} MB")
        with col3:
            st.info(f"**Type:** {uploaded_video.type}")

        # Processing options
        st.markdown("### Processing Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            frame_interval = st.number_input(
                "Frame Interval",
                min_value=1,
                max_value=300,
                value=30,
                help="Extract every Nth frame (lower = more frames)"
            )

        with col2:
            auto_predict = st.checkbox(
                "Auto-run predictions",
                value=True,
                help="Automatically analyze frames after extraction"
            )

        with col3:
            quality_mode = st.selectbox(
                "Quality Mode:",
                ["Standard", "High Quality", "Fast"],
                help="Processing quality vs speed trade-off"
            )

        # Processing preview
        if frame_interval > 0:
            # Estimate processing time and frame count
            estimated_frames = max(1, int(file_size_mb * 30 / frame_interval))  # Rough estimate
            estimated_time = estimated_frames * 0.1  # Rough estimate

            st.info(f"📊 Estimated: ~{estimated_frames} frames, ~{estimated_time:.1f}s processing time")

        # Process button
        if st.button("🚀 Upload & Process Video", type="primary", use_container_width=True):
            process_uploaded_video(uploaded_video, frame_interval, auto_predict, quality_mode)


def process_uploaded_video(uploaded_video, frame_interval: int, auto_predict: bool, quality_mode: str):
    """Process the uploaded video"""
    with st.spinner("Uploading and processing video..."):
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Upload video
            status_text.text("📤 Uploading video...")
            progress_bar.progress(0.2)

            result = st.session_state.api_client.upload_video(
                uploaded_video.getvalue(),
                uploaded_video.name,
                frame_interval
            )

            if not result["success"]:
                st.error(f"❌ Upload failed: {result['error']}")
                return

            # Step 2: Frame extraction completed
            status_text.text("🎞️ Frames extracted successfully!")
            progress_bar.progress(0.6)

            video_data = result["data"]
            st.session_state.current_video = {
                'name': video_data['video_name'],
                'frames_extracted': video_data['total_frames_extracted'],
                'upload_path': video_data['upload_path']
            }

            # Show extraction results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Video ID", video_data['video_id'][:8] + "...")
            with col2:
                st.metric("Frames Extracted", video_data['total_frames_extracted'])
            with col3:
                st.metric("Status", "✅ Ready")

            progress_bar.progress(0.8)

            # Step 3: Auto-predict if enabled
            if auto_predict:
                status_text.text("🤖 Running AI predictions...")

                prediction_result = st.session_state.api_client.predict_frames(video_data['video_name'])

                if prediction_result["success"]:
                    st.session_state.video_predictions = prediction_result["data"]

                    # Show prediction summary
                    pred_data = prediction_result["data"]
                    st.success(f"✅ Analysis complete! Generated {pred_data['total_predictions']} predictions")

                    # Display prediction breakdown
                    if pred_data['predictions_by_class']:
                        st.markdown("**Prediction Summary:**")
                        for class_name, count in pred_data['predictions_by_class'].items():
                            percentage = (count / pred_data['total_predictions']) * 100
                            st.write(f"- {class_name.title()}: {count} ({percentage:.1f}%)")

                else:
                    st.warning(f"⚠️ Prediction failed: {prediction_result['error']}")

            progress_bar.progress(1.0)
            status_text.text("🎉 Processing complete!")

            # Success actions
            st.balloons()

            # Quick action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📊 View Predictions"):
                    st.session_state.page = "feedback"
                    st.rerun()

            with col2:
                if st.button("📋 Export Results"):
                    export_video_analysis()

            with col3:
                if st.button("🔄 Process Another"):
                    SessionManager.clear_video_data()
                    st.rerun()

        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()


def render_video_management():
    """Render video management interface"""
    st.markdown("### Manage Processed Videos")

    # Get list of videos
    result = st.session_state.api_client.list_videos()

    if result["success"]:
        videos_data = result["data"]

        if videos_data["videos"]:
            # Display videos in a table
            df_data = []
            for video in videos_data["videos"]:
                df_data.append({
                    "Video Name": video["video_name"],
                    "Frames": video["frame_count"],
                    "Has Predictions": "✅" if video["has_predictions"] else "❌",
                    "Predictions Count": video["prediction_count"] if video["has_predictions"] else 0,
                    "Status": "Ready for Feedback" if video["has_predictions"] else "Needs Analysis"
                })

            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)

            # Video selection for actions
            st.markdown("### Video Actions")

            selected_video, video_data = UIComponents.render_video_selector()

            if selected_video:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if video_data["has_predictions"]:
                        if st.button("📊 View Predictions"):
                            st.session_state.selected_video_for_feedback = selected_video
                            st.session_state.page = "feedback"
                            st.rerun()
                    else:
                        if st.button("🤖 Run Analysis"):
                            run_analysis_on_video(selected_video)

                with col2:
                    if st.button("📋 Export Data"):
                        export_video_data(selected_video)

                with col3:
                    if st.button("🔄 Re-analyze"):
                        reanalyze_video(selected_video)

                with col4:
                    if st.button("🗑️ Delete", type="secondary"):
                        if st.checkbox(f"Confirm delete {selected_video}"):
                            delete_video_data(selected_video)
        else:
            st.info("📁 No processed videos found. Upload a video to get started!")
    else:
        st.error(f"❌ Error loading videos: {result['error']}")


def render_advanced_analysis():
    """Render advanced analysis interface"""
    st.markdown("### Advanced Video Analysis")

    # Batch processing section
    with st.expander("🔄 Batch Processing", expanded=False):
        st.markdown("Process multiple videos with the same settings")

        batch_files = st.file_uploader(
            "Select multiple video files",
            type=VALID_VIDEO_FORMATS,
            accept_multiple_files=True
        )

        if batch_files:
            col1, col2 = st.columns(2)
            with col1:
                batch_frame_interval = st.number_input("Frame Interval", 1, 300, 30, key="batch_interval")
            with col2:
                batch_auto_predict = st.checkbox("Auto-predict all", True, key="batch_predict")

            if st.button("🚀 Process Batch"):
                process_batch_videos(batch_files, batch_frame_interval, batch_auto_predict)

    # Analysis comparison
    with st.expander("📊 Compare Analyses", expanded=False):
        st.markdown("Compare prediction results across multiple videos")

        result = st.session_state.api_client.list_videos()
        if result["success"] and result["data"]["videos"]:
            videos_with_predictions = [v for v in result["data"]["videos"] if v["has_predictions"]]

            if len(videos_with_predictions) >= 2:
                selected_videos = st.multiselect(
                    "Select videos to compare:",
                    [v["video_name"] for v in videos_with_predictions]
                )

                if len(selected_videos) >= 2:
                    if st.button("📈 Generate Comparison"):
                        generate_video_comparison(selected_videos)
            else:
                st.info("Need at least 2 analyzed videos for comparison")


def render_expert_feedback():
    """Render expert feedback interface"""
    st.markdown('<h2 class="section-header">📝 Expert Feedback Interface</h2>', unsafe_allow_html=True)

    # Create tabs for different feedback modes
    tab1, tab2, tab3 = st.tabs(["🎯 Review Predictions", "📤 Manual Upload", "📊 Batch Review"])

    with tab1:
        render_prediction_review()

    with tab2:
        render_manual_feedback()

    with tab3:
        render_batch_feedback()


def render_prediction_review():
    """Render prediction review interface"""
    st.markdown("### Review AI Predictions")

    # Video selection
    selected_video, video_data = UIComponents.render_video_selector()
    print("Video Selected" + selected_video)
    if not selected_video:
        st.info("👆 Select a video to review predictions")
        return

    if not video_data["has_predictions"]:
        st.warning("⚠️ Selected video has no predictions. Run analysis first.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🤖 Run Analysis Now"):
                run_analysis_on_video(selected_video)
        return

    # Load predictions with filters
    st.markdown("### Filter Predictions")
    include_normal, class_filter, confidence_threshold = UIComponents.render_prediction_filters()

    # Get filtered predictions
    result = st.session_state.api_client.get_predictions(
        selected_video,
        include_normal,
        class_filter if class_filter != "All" else None,
        confidence_threshold
    )

    if result["success"]:
        predictions_data = result["data"]
        predictions = predictions_data["predictions"]

        if predictions:
            # Display summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", predictions_data["total_predictions"])
            with col2:
                st.metric("Filtered Results", len(predictions))
            with col3:
                defect_count = len([p for p in predictions if p["predicted_class"].lower() != "normal"])
                st.metric("Potential Defects", defect_count)
            with col4:
                avg_confidence = sum(float(p["confidence"]) for p in predictions) / len(predictions)
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")

            # Sort and display options
            st.markdown("### Review Options")
            col1, col2, col3 = st.columns(3)

            with col1:
                sort_by = st.selectbox(
                    "Sort by:",
                    ["Frame Order", "Confidence (Low to High)", "Confidence (High to Low)", "Class"]
                )

            with col2:
                items_per_page = st.selectbox("Items per page:", [5, 10, 20, 50], index=1)

            with col3:
                review_mode = st.selectbox(
                    "Review Mode:",
                    ["Standard", "Quick Review", "Detailed Analysis"]
                )

            # Apply sorting
            if sort_by == "Confidence (Low to High)":
                predictions = sorted(predictions, key=lambda x: float(x['confidence']))
            elif sort_by == "Confidence (High to Low)":
                predictions = sorted(predictions, key=lambda x: float(x['confidence']), reverse=True)
            elif sort_by == "Class":
                predictions = sorted(predictions, key=lambda x: x['predicted_class'])

            # Pagination
            total_pages = len(predictions) // items_per_page + (1 if len(predictions) % items_per_page > 0 else 0)

            if total_pages > 1:
                page = st.selectbox("Page:", range(1, total_pages + 1))
                start_idx = (page - 1) * items_per_page
                end_idx = start_idx + items_per_page
                current_predictions = predictions[start_idx:end_idx]
            else:
                current_predictions = predictions
                start_idx = 0

            # Display predictions based on review mode
            if review_mode == "Quick Review":
                render_quick_review_mode(current_predictions, start_idx)
            elif review_mode == "Detailed Analysis":
                render_detailed_analysis_mode(current_predictions, start_idx)
            else:
                render_standard_review_mode(current_predictions, start_idx)

        else:
            st.info("🔍 No predictions match the current filter criteria")
    else:
        st.error(f"❌ Error loading predictions: {result['error']}")


def render_standard_review_mode(predictions: List[Dict], start_idx: int):
    """Render standard review mode"""
    for i, prediction in enumerate(predictions):
        UIComponents.render_prediction_card(prediction, start_idx + i)


def render_quick_review_mode(predictions: List[Dict], start_idx: int):
    """Render quick review mode for fast feedback"""
    st.markdown("### Quick Review Mode")
    st.info("💡 Quick review mode for fast feedback on multiple predictions")

    # Batch actions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Confirm All Visible"):
            confirm_all_predictions(predictions)
    with col2:
        if st.button("🔄 Mark All for Review"):
            mark_all_for_review(predictions)
    with col3:
        if st.button("📊 Quick Stats"):
            show_quick_stats(predictions)

    # Compact prediction display
    for i, prediction in enumerate(predictions):
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

            with col1:
                confidence = float(prediction['confidence'])
                st.write(f"**Frame {start_idx + i + 1}**")
                st.write(f"{prediction['predicted_class'].title()} ({confidence:.1%})")

            with col2:
                try:
                    image_url = f"{API_BASE_URL}/{prediction['image_path']}"
                    st.image(image_url, width=100)
                except:
                    st.write("Image unavailable")

            with col3:
                # Quick feedback buttons
                if st.button("✅", key=f"confirm_{prediction['frame_id']}", help="Confirm"):
                    quick_confirm_prediction(prediction)
                if st.button("❌", key=f"reject_{prediction['frame_id']}", help="Correct"):
                    show_quick_correction_form(prediction)
                if st.button("❓", key=f"review_{prediction['frame_id']}", help="Flag for Review"):
                    flag_for_detailed_review(prediction)

            with col4:
                # Status indicator
                if prediction['predicted_class'].lower() != 'normal':
                    if confidence > 0.8:
                        st.write("🔴 High")
                    elif confidence > 0.6:
                        st.write("🟡 Med")
                    else:
                        st.write("🟠 Low")
                else:
                    st.write("🟢 Normal")

            st.divider()


def render_detailed_analysis_mode(predictions: List[Dict], start_idx: int):
    """Render detailed analysis mode with comprehensive feedback"""
    st.markdown("### Detailed Analysis Mode")
    st.info("🔬 Comprehensive analysis mode with detailed feedback forms")

    for i, prediction in enumerate(predictions):
        with st.expander(
                f"🔍 Detailed Analysis - Frame {start_idx + i + 1}: {prediction['predicted_class'].title()}",
                expanded=True
        ):
            col1, col2 = st.columns([1, 1])

            with col1:
                # Enhanced image display
                try:
                    image_url = f"{API_BASE_URL}/{prediction['image_path']}"
                    st.image(image_url, caption=f"Frame {start_idx + i + 1}", use_container_width=True)

                    # Image analysis tools
                    st.markdown("**Image Analysis Tools:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("🔍 Enhance", key=f"enhance_{prediction['frame_id']}"):
                            st.info("Image enhancement feature coming soon")
                    with col_b:
                        if st.button("📏 Measure", key=f"measure_{prediction['frame_id']}"):
                            st.info("Measurement tools coming soon")

                except:
                    st.error("Could not load image")

            with col2:
                # Comprehensive feedback form
                with st.form(f"detailed_feedback_{prediction['frame_id']}"):
                    st.markdown("**Comprehensive Expert Analysis**")

                    # Primary classification
                    expert_class = st.selectbox(
                        "Primary Classification:",
                        CLASSIFICATION_CLASSES,
                        key=f"detailed_class_{prediction['frame_id']}"
                    )

                    # Secondary classification (if applicable)
                    if expert_class != "normal":
                        secondary_class = st.multiselect(
                            "Secondary Issues (if any):",
                            [c for c in CLASSIFICATION_CLASSES if c != expert_class],
                            key=f"secondary_{prediction['frame_id']}"
                        )

                    # Detailed metrics
                    col_1, col_2 = st.columns(2)
                    with col_1:
                        confidence = st.slider("Confidence:", 0.0, 1.0, 0.9,
                                               key=f"detailed_conf_{prediction['frame_id']}")
                        severity = st.selectbox("Severity:", ["Low", "Medium", "High", "Critical"],
                                                key=f"detailed_sev_{prediction['frame_id']}")

                    with col_2:
                        urgency = st.selectbox("Urgency:", ["Low", "Medium", "High", "Immediate"],
                                               key=f"detailed_urg_{prediction['frame_id']}")
                        location_quality = st.selectbox("Location Quality:", ["Poor", "Fair", "Good", "Excellent"],
                                                        key=f"location_qual_{prediction['frame_id']}")

                    # Detailed descriptions
                    defect_description = st.text_area(
                        "Defect Description:",
                        placeholder="Detailed description of the defect, including size, shape, location...",
                        key=f"detailed_desc_{prediction['frame_id']}"
                    )

                    recommended_action = st.text_area(
                        "Recommended Action:",
                        placeholder="Specific actions recommended based on this finding...",
                        key=f"detailed_action_{prediction['frame_id']}"
                    )

                    # Additional metadata
                    with st.expander("Additional Metadata"):
                        asset_id = st.text_input("Asset/Pipe ID:", key=f"asset_{prediction['frame_id']}")
                        inspector_notes = st.text_area("Inspector Notes:", key=f"inspector_{prediction['frame_id']}")
                        follow_up_required = st.checkbox("Follow-up Required", key=f"followup_{prediction['frame_id']}")

                        if follow_up_required:
                            follow_up_date = st.date_input("Follow-up Date:",
                                                           key=f"followup_date_{prediction['frame_id']}")

                    # Submit comprehensive feedback
                    if st.form_submit_button("Submit Detailed Analysis", type="primary"):
                        submit_detailed_feedback(prediction, {
                            'expert_class': expert_class,
                            'confidence': confidence,
                            'severity': severity,
                            'urgency': urgency,
                            'description': defect_description,
                            'recommended_action': recommended_action,
                            'asset_id': asset_id,
                            'inspector_notes': inspector_notes,
                            'follow_up_required': follow_up_required
                        })


def render_manual_feedback():
    """Render manual image upload feedback interface"""
    st.markdown("### Manual Image Upload")
    st.info("💡 Upload individual images for expert classification")

    # Image upload
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=VALID_IMAGE_FORMATS,
        help=f"Supported formats: {', '.join(VALID_IMAGE_FORMATS).upper()}"
    )

    if uploaded_image:
        # Display image
        image = Image.open(uploaded_image)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Image info
            st.markdown("**Image Information:**")
            st.write(f"- **Filename:** {uploaded_image.name}")
            st.write(f"- **Size:** {uploaded_image.size}")
            st.write(f"- **Format:** {image.format}")
            st.write(f"- **Mode:** {image.mode}")

        with col2:
            # Manual classification form
            with st.form("manual_feedback_form"):
                st.markdown("**Expert Classification**")

                # Basic classification
                frame_id = st.text_input(
                    "Frame/Image ID:",
                    value=f"manual_{int(time.time())}",
                    help="Unique identifier for this image"
                )

                model_prediction = st.selectbox(
                    "Model Prediction (if known):",
                    ["Unknown"] + CLASSIFICATION_CLASSES,
                    help="What did the AI model predict?"
                )

                expert_classification = st.selectbox(
                    "Expert Classification:",
                    CLASSIFICATION_CLASSES,
                    help="Your expert classification"
                )

                # Confidence and additional info
                confidence_score = st.slider("Confidence Level:", 0.0, 1.0, 0.8)

                # Additional details
                with st.expander("Additional Details"):
                    asset_location = st.text_input("Asset/Location ID:")
                    severity_level = st.selectbox("Severity:", ["Low", "Medium", "High", "Critical"])
                    priority_level = st.selectbox("Priority:", ["Low", "Medium", "High", "Urgent"])

                expert_notes = st.text_area(
                    "Expert Notes:",
                    placeholder="Detailed notes about the classification, defect characteristics, recommended actions...",
                    height=150
                )

                if st.form_submit_button("Submit Manual Feedback", type="primary"):
                    submit_manual_feedback(
                        uploaded_image, frame_id, model_prediction, expert_classification,
                        confidence_score, expert_notes, asset_location, severity_level, priority_level
                    )


def render_batch_feedback():
    """Render batch feedback interface"""
    st.markdown("### Batch Feedback Processing")
    st.info("📦 Process multiple images or predictions in batch mode")

    # Batch processing options
    batch_mode = st.radio(
        "Batch Processing Mode:",
        ["Upload Multiple Images", "Process Prediction Batch", "Import from CSV"]
    )

    if batch_mode == "Upload Multiple Images":
        render_batch_image_upload()
    elif batch_mode == "Process Prediction Batch":
        render_batch_prediction_processing()
    else:
        render_csv_import()


def render_batch_image_upload():
    """Render batch image upload interface"""
    st.markdown("#### Upload Multiple Images")

    uploaded_files = st.file_uploader(
        "Select multiple image files",
        type=VALID_IMAGE_FORMATS,
        accept_multiple_files=True,
        help="Upload multiple images for batch classification"
    )

    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} images selected")

        # Batch settings
        col1, col2 = st.columns(2)

        with col1:
            default_classification = st.selectbox(
                "Default Classification:",
                CLASSIFICATION_CLASSES,
                help="Default classification for all images (can be changed individually)"
            )

        with col2:
            default_confidence = st.slider(
                "Default Confidence:",
                0.0, 1.0, 0.8,
                help="Default confidence level"
            )

        # Preview and process
        if st.button("🔍 Preview Batch"):
            preview_batch_images(uploaded_files, default_classification, default_confidence)

        if st.button("🚀 Process Batch", type="primary"):
            process_batch_images(uploaded_files, default_classification, default_confidence)


def render_analytics():
    """Render analytics and reporting interface"""
    st.markdown('<h2 class="section-header">📊 Analytics & Reporting</h2>', unsafe_allow_html=True)

    # Update stats
    SessionManager.update_feedback_stats()

    if not st.session_state.feedback_stats or st.session_state.feedback_stats['total_samples'] == 0:
        st.info("📈 No analytics available yet. Submit some feedback to see analytics!")
        return

    stats = st.session_state.feedback_stats

    # Create tabs for different analytics views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🎯 Performance", "📊 Trends", "📋 Reports"])

    with tab1:
        render_analytics_overview(stats)

    with tab2:
        render_performance_analytics(stats)

    with tab3:
        render_trend_analytics(stats)

    with tab4:
        render_reporting_interface(stats)


def render_analytics_overview(stats: Dict):
    """Render analytics overview"""
    st.markdown("### System Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Samples", stats['total_samples'])

    with col2:
        st.metric("Corrections Made", stats['corrections_count'])

    with col3:
        correction_rate = stats['correction_rate'] * 100
        st.metric("Correction Rate", f"{correction_rate:.1f}%")

    with col4:
        if stats['total_samples'] > 0:
            accuracy = ((stats['total_samples'] - stats['corrections_count']) / stats['total_samples']) * 100
            st.metric("Model Accuracy", f"{accuracy:.1f}%")

    # Visual representations
    if stats['by_classification']:
        col1, col2 = st.columns(2)

        with col1:
            # Class distribution pie chart
            fig_pie = px.pie(
                values=list(stats['by_classification'].values()),
                names=[name.title() for name in stats['by_classification'].keys()],
                title="Sample Distribution by Classification"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Class distribution bar chart
            fig_bar = px.bar(
                x=[name.title() for name in stats['by_classification'].keys()],
                y=list(stats['by_classification'].values()),
                title="Sample Count by Classification"
            )
            fig_bar.update_layout(xaxis_title="Classification", yaxis_title="Count")
            st.plotly_chart(fig_bar, use_container_width=True)

    # Progress tracking
    st.markdown("### Training Progress")
    progress = min(stats['total_samples'] / stats['retrain_threshold'], 1.0)
    st.progress(progress)

    remaining = max(0, stats['retrain_threshold'] - stats['total_samples'])
    if remaining == 0:
        st.success("🎯 Ready for retraining!")
    else:
        st.info(f"📊 {remaining} more samples needed for retraining")


def render_model_management():
    """Render model management interface"""
    st.markdown('<h2 class="section-header">🤖 Model Management</h2>', unsafe_allow_html=True)

    # Create tabs for different management functions
    tab1, tab2, tab3 = st.tabs(["🔄 Retraining", "📊 Model Status", "⚙️ Configuration"])

    with tab1:
        render_retraining_interface()

    with tab2:
        render_model_status()

    with tab3:
        render_model_configuration()


def render_retraining_interface():
    """Render model retraining interface"""
    st.markdown("### Model Retraining")

    # Get current status
    SessionManager.update_feedback_stats()

    if st.session_state.feedback_stats:
        stats = st.session_state.feedback_stats

        # Retraining status
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Samples Collected", stats['total_samples'])

        with col2:
            st.metric("Threshold", stats['retrain_threshold'])

        with col3:
            ready = stats['ready_for_retrain']
            st.metric("Status", "Ready" if ready else "Collecting")

        # Progress bar
        progress = min(stats['total_samples'] / stats['retrain_threshold'], 1.0)
        st.progress(progress)

        # Retraining options
        if ready:
            st.success("✅ System is ready for retraining!")

            with st.expander("⚙️ Retraining Options", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    backup_current = st.checkbox("Backup current model", value=True)
                    use_all_data = st.checkbox("Use all available data", value=True)

                with col2:
                    validation_split = st.slider("Validation split:", 0.1, 0.3, 0.2)
                    max_epochs = st.number_input("Max epochs:", 1, 100, 10)

                # Advanced options
                with st.expander("Advanced Options"):
                    learning_rate = st.number_input("Learning rate:", 0.0001, 0.1, 0.001, format="%.4f")
                    batch_size = st.selectbox("Batch size:", [16, 32, 64, 128], index=1)
                    early_stopping = st.checkbox("Early stopping", value=True)

            # Retraining button
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🚀 Start Retraining", type="primary", use_container_width=True):
                    start_model_retraining(backup_current, use_all_data, validation_split, max_epochs)

            with col2:
                if st.button("📊 Prepare Training Data", use_container_width=True):
                    prepare_training_data()

        else:
            remaining = stats['retrain_threshold'] - stats['total_samples']
            st.warning(f"⏳ Need {remaining} more samples before retraining")

            # Show what's needed
            st.markdown("### What's needed:")
            st.write(f"- Current samples: {stats['total_samples']}")
            st.write(f"- Required samples: {stats['retrain_threshold']}")
            st.write(f"- Remaining: {remaining}")

            if st.button("📊 View Collection Progress"):
                show_collection_progress(stats)


# Helper functions for the enhanced UI

def run_analysis_on_video(video_name: str):
    """Run analysis on a specific video"""
    with st.spinner(f"Running analysis on {video_name}..."):
        result = st.session_state.api_client.predict_frames(video_name)

        if result["success"]:
            st.success("✅ Analysis completed!")
            st.rerun()
        else:
            st.error(f"❌ Analysis failed: {result['error']}")


def export_video_data(video_name: str):
    """Export video analysis data"""
    # Implementation for exporting video data
    st.info("📊 Export functionality coming soon!")


def process_batch_videos(batch_files, frame_interval: int, auto_predict: bool):
    """Process multiple videos in batch"""
    total_files = len(batch_files)
    progress_bar = st.progress(0)
    results = []

    for i, file in enumerate(batch_files):
        st.write(f"Processing {file.name}...")

        # Process individual video
        result = st.session_state.api_client.upload_video(
            file.getvalue(),
            file.name,
            frame_interval
        )

        results.append({
            'filename': file.name,
            'success': result['success'],
            'details': result.get('data', result.get('error'))
        })

        # Update progress
        progress_bar.progress((i + 1) / total_files)

    # Show results
    st.markdown("### Batch Processing Results")
    for result in results:
        if result['success']:
            st.success(f"✅ {result['filename']} - Processed successfully")
        else:
            st.error(f"❌ {result['filename']} - Failed: {result['details']}")


def start_model_retraining(backup_current: bool, use_all_data: bool,
                           validation_split: float, max_epochs: int):
    """Start the model retraining process"""
    with st.spinner("🔄 Starting model retraining... This may take several minutes."):
        result = st.session_state.api_client.retrain_model()

        if result["success"]:
            st.success("🎉 Model retrained successfully!")
            st.balloons()

            # Show retraining results
            retraining_data = result["data"]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Samples", retraining_data.get('active_learning_samples', 0))
            with col2:
                st.metric("Used Active Learning", "Yes" if retraining_data.get('used_active_learning') else "No")

            # Update session state
            SessionManager.update_feedback_stats()

        else:
            st.error(f"❌ Retraining failed: {result['error']}")


def main():
    """Main application entry point"""
    # Initialize session
    SessionManager.init_session()

    # Render status indicator
    UIComponents.render_status_indicator()

    # Main navigation
    st.sidebar.markdown("---")
    st.sidebar.header("🧭 Navigation")

    # Navigation menu
    page = st.sidebar.selectbox(
        "Choose a section:",
        [
            "🏠 Dashboard",
            "🎥 Video Processing",
            "📝 Expert Feedback",
            "📊 Analytics",
            "🤖 Model Management"
        ]
    )

    # Store current page in session
    st.session_state.page = page

    # Route to appropriate interface
    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "🎥 Video Processing":
        render_video_processing()
    elif page == "📝 Expert Feedback":
        render_expert_feedback()
    elif page == "📊 Analytics":
        render_analytics()
    elif page == "🤖 Model Management":
        render_model_management()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**CBM Expert System v2.0**")
    st.sidebar.caption("Enhanced AI-powered condition monitoring")


if __name__ == "__main__":
    main()