import streamlit as st
import requests
import json
from PIL import Image
import io
import pandas as pd
from datetime import datetime
import time

# Configuration
API_BASE_URL = "http://localhost:8000"  # Adjust to your FastAPI server URL


def expert_feedback_interface():
    """
    Main function for expert feedback interface in Streamlit
    """
    st.title("🔧 Expert Feedback System")
    st.write("Provide expert annotations to improve the CBM model through active learning")

    # Sidebar for navigation
    st.sidebar.header("Navigation")
    mode = st.sidebar.selectbox(
        "Choose Mode:",
        ["Submit Feedback", "View Statistics", "Manage Retraining"]
    )

    if mode == "Submit Feedback":
        submit_feedback_interface()
    elif mode == "View Statistics":
        view_statistics_interface()
    elif mode == "Manage Retraining":
        manage_retraining_interface()


def submit_feedback_interface():
    """
    Interface for submitting expert feedback
    """
    st.header("📝 Submit Expert Feedback")

    # Option to choose between uploaded image or prediction results
    feedback_source = st.radio(
        "Feedback Source:",
        ["Upload New Image", "From Prediction Results"]
    )

    if feedback_source == "Upload New Image":
        upload_image_feedback()
    else:
        prediction_results_feedback()


def upload_image_feedback():
    """
    Handle feedback submission for uploaded images
    """
    st.subheader("Upload Image for Feedback")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image that needs expert classification"
    )

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Feedback form
        with st.form("feedback_form"):
            st.subheader("Expert Classification")

            col1, col2 = st.columns(2)

            with col1:
                # Generate a frame ID
                frame_id = st.text_input(
                    "Frame ID",
                    value=f"manual_{int(time.time())}",
                    help="Unique identifier for this image"
                )

                model_prediction = st.selectbox(
                    "Model Prediction (if any)",
                    ["normal", "crack", "corrosion", "leakage", "unknown"],
                    index=4,
                    help="What did the model predict for this image?"
                )

                expert_classification = st.selectbox(
                    "Expert Classification *",
                    ["normal", "crack", "corrosion", "leakage"],
                    help="Your expert classification of this image"
                )

            with col2:
                confidence_score = st.slider(
                    "Confidence Score",
                    0.0, 1.0, 0.8,
                    help="How confident are you in this classification?"
                )

                pipe_id = st.text_input(
                    "Pipe ID (optional)",
                    help="Identifier of the pipe/asset in the image"
                )

            expert_notes = st.text_area(
                "Expert Notes (optional)",
                help="Additional notes or explanations about your classification"
            )

            submitted = st.form_submit_button("Submit Feedback")

            if submitted:
                submit_expert_feedback_to_api(
                    uploaded_file, frame_id, model_prediction,
                    expert_classification, confidence_score,
                    expert_notes, pipe_id
                )


def prediction_results_feedback():
    """
    Handle feedback submission based on prediction results
    """
    st.subheader("Feedback on Model Predictions")

    # Button to get predictions
    if st.button("Get Latest Predictions"):
        get_and_display_predictions()

    # Check if predictions are stored in session state
    if 'predictions' in st.session_state:
        display_predictions_for_feedback()


def get_and_display_predictions():
    """
    Fetch predictions from the API and store them in session state
    """
    try:
        with st.spinner("Fetching predictions..."):
            response = requests.post(f"{API_BASE_URL}/predict_frames", json={})

        if response.status_code == 200:
            data = response.json()
            predictions = json.loads(data['results'])
            st.session_state.predictions = predictions
            st.success(f"Loaded {len(predictions)} predictions")
        else:
            st.error(f"Error fetching predictions: {response.text}")

    except Exception as e:
        st.error(f"Connection error: {str(e)}")


def display_predictions_for_feedback():
    """
    Display predictions and allow expert feedback
    """
    predictions = st.session_state.predictions

    # Filter and pagination
    col1, col2, col3 = st.columns(3)

    with col1:
        filter_class = st.selectbox(
            "Filter by predicted class:",
            ["All"] + list(set([p['label'] for p in predictions]))
        )

    with col2:
        items_per_page = st.selectbox("Items per page:", [5, 10, 20], index=1)

    with col3:
        if filter_class != "All":
            filtered_predictions = [p for p in predictions if p['label'] == filter_class]
        else:
            filtered_predictions = predictions

        total_pages = len(filtered_predictions) // items_per_page + (
            1 if len(filtered_predictions) % items_per_page > 0 else 0)
        page = st.selectbox("Page:", range(1, total_pages + 1))

    # Display predictions for current page
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    current_predictions = filtered_predictions[start_idx:end_idx]

    for i, pred in enumerate(current_predictions):
        with st.expander(f"Prediction {start_idx + i + 1}: {pred['label']} (Confidence: {float(pred['confidence']):.3f})"):

            col1, col2 = st.columns([1, 2])

            with col1:
                # Display image if path exists
                try:
                    # Construct the full URL for the image
                    image_url = f"{API_BASE_URL}/{pred['path']}"
                    st.image(image_url, caption=f"Frame: {pred['path']}")
                except:
                    st.write(f"Image path: {pred['path']}")

            with col2:
                # Feedback form for this prediction
                with st.form(f"feedback_form_{pred['frame_id']}"):
                    expert_classification = st.selectbox(
                        "Expert Classification",
                        ["normal", "crack", "corrosion", "leakage"],
                        key=f"expert_class_{pred['frame_id']}"
                    )

                    confidence_score = st.slider(
                        "Confidence",
                        0.0, 1.0, 0.8,
                        key=f"confidence_{pred['frame_id']}"
                    )

                    expert_notes = st.text_area(
                        "Notes",
                        key=f"notes_{pred['frame_id']}",
                        height=100
                    )

                    if st.form_submit_button("Submit Correction"):
                        # Download image and submit feedback
                        submit_prediction_feedback(
                            pred, expert_classification,
                            confidence_score, expert_notes
                        )


def submit_expert_feedback_to_api(uploaded_file, frame_id, model_prediction,
                                  expert_classification, confidence_score,
                                  expert_notes, pipe_id):
    """
    Submit feedback to the FastAPI backend
    """
    try:
        # Prepare the files and data for the request
        files = {
            'frame_image': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        data = {
            'frame_id': frame_id,
            'model_prediction': model_prediction,
            'expert_classification': expert_classification,
            'confidence_score': confidence_score,
            'expert_notes': expert_notes or "",
            'pipe_id': pipe_id or ""
        }

        with st.spinner("Submitting feedback..."):
            response = requests.post(
                f"{API_BASE_URL}/feedback/expert",
                files=files,
                data=data
            )

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Feedback submitted successfully!")

            # Display feedback details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Feedback ID", result['feedback_id'][:8] + "...")
            with col2:
                st.metric("Total Samples", result['samples_collected'])
            with col3:
                st.metric("Ready for Retrain", "Yes" if result['ready_for_retrain'] else "No")

            if result['ready_for_retrain']:
                st.info("🔄 Enough samples collected for model retraining!")

        else:
            st.error(f"Error submitting feedback: {response.text}")

    except Exception as e:
        st.error(f"Connection error: {str(e)}")


def submit_prediction_feedback(prediction, expert_classification, confidence_score, expert_notes):
    """
    Submit feedback for a prediction result
    """
    try:
        # Download the image from the prediction path
        image_url = f"{API_BASE_URL}/{prediction['path']}"
        image_response = requests.get(image_url)

        if image_response.status_code != 200:
            st.error("Could not download image for feedback")
            return

        files = {
            'frame_image': ('frame.jpg', image_response.content, 'image/jpeg')
        }

        data = {
            'frame_id': prediction['frame_id'],
            'model_prediction': prediction['label'],
            'expert_classification': expert_classification,
            'confidence_score': confidence_score,
            'expert_notes': expert_notes or ""
        }

        with st.spinner("Submitting feedback..."):
            response = requests.post(
                f"{API_BASE_URL}/feedback/expert",
                files=files,
                data=data
            )

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Feedback submitted!")

            # Show brief confirmation
            if result['ready_for_retrain']:
                st.info("🔄 Ready for retraining!")
        else:
            st.error(f"Error: {response.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")


def view_statistics_interface():
    """
    Display statistics about collected feedback
    """
    st.header("📊 Feedback Statistics")

    if st.button("Refresh Statistics"):
        get_feedback_statistics()

    # Auto-load statistics on page load
    if 'stats_loaded' not in st.session_state:
        get_feedback_statistics()
        st.session_state.stats_loaded = True


def get_feedback_statistics():
    """
    Fetch and display feedback statistics
    """
    try:
        response = requests.get(f"{API_BASE_URL}/feedback/stats")

        if response.status_code == 200:
            stats = response.json()

            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Samples", stats['total_samples'])
            with col2:
                st.metric("Corrections", stats['corrections_count'])
            with col3:
                st.metric("Correction Rate", f"{stats['correction_rate']:.1%}")
            with col4:
                st.metric("Ready for Retrain", "Yes" if stats['ready_for_retrain'] else "No")

            # Progress bar for retraining threshold
            st.subheader("Progress to Retraining")
            progress = min(stats['total_samples'] / stats['retrain_threshold'], 1.0)
            st.progress(progress)
            st.write(f"{stats['total_samples']} / {stats['retrain_threshold']} samples collected")

            # Class distribution
            if stats['by_classification']:
                st.subheader("Sample Distribution by Class")

                # Create DataFrame for better display
                class_data = []
                for class_name, count in stats['by_classification'].items():
                    percentage = (count / stats['total_samples']) * 100
                    class_data.append({
                        'Class': class_name.title(),
                        'Count': count,
                        'Percentage': f"{percentage:.1f}%"
                    })

                df = pd.DataFrame(class_data)
                st.dataframe(df, use_container_width=True)

                # Bar chart
                st.bar_chart(stats['by_classification'])

        else:
            st.error(f"Error fetching statistics: {response.text}")

    except Exception as e:
        st.error(f"Connection error: {str(e)}")


def manage_retraining_interface():
    """
    Interface for managing model retraining
    """
    st.header("🔄 Model Retraining Management")

    # Get current statistics
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Check Retraining Status"):
            check_retraining_status()

    with col2:
        if st.button("Prepare for Retraining"):
            prepare_retraining()

    st.markdown("---")

    # Retraining section
    st.subheader("Start Retraining")
    st.warning("⚠️ Retraining will update the current model. This process may take several minutes.")

    if st.button("🚀 Start Retraining", type="primary"):
        start_retraining()

    st.markdown("---")

    # Danger zone
    with st.expander("⚠️ Danger Zone", expanded=False):
        st.error("These actions cannot be undone!")

        if st.button("🗑️ Reset All Feedback Data"):
            if st.checkbox("I understand this will delete all feedback data"):
                reset_feedback_data()


def check_retraining_status():
    """
    Check if the system is ready for retraining
    """
    try:
        response = requests.get(f"{API_BASE_URL}/feedback/stats")

        if response.status_code == 200:
            stats = response.json()

            if stats['ready_for_retrain']:
                st.success("✅ System is ready for retraining!")
                st.info(f"Collected {stats['total_samples']} samples (threshold: {stats['retrain_threshold']})")
            else:
                needed = stats['retrain_threshold'] - stats['total_samples']
                st.warning(f"⏳ Need {needed} more samples before retraining")

        else:
            st.error(f"Error checking status: {response.text}")

    except Exception as e:
        st.error(f"Connection error: {str(e)}")


def prepare_retraining():
    """
    Prepare the system for retraining
    """
    try:
        with st.spinner("Preparing retraining data..."):
            response = requests.post(f"{API_BASE_URL}/feedback/prepare_retrain")

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Retraining data prepared!")

            # Show preparation details
            st.json(result)

        else:
            st.error(f"Error preparing retraining: {response.text}")

    except Exception as e:
        st.error(f"Connection error: {str(e)}")


def start_retraining():
    """
    Start the model retraining process
    """
    try:
        with st.spinner("Retraining model... This may take several minutes."):
            response = requests.post(f"{API_BASE_URL}/retrain")

        if response.status_code == 200:
            result = response.json()
            st.success("🎉 Model retrained successfully!")

            # Show retraining details
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Active Learning Samples", result['active_learning_samples'])
            with col2:
                st.metric("Used Active Learning", "Yes" if result['used_active_learning'] else "No")

            st.balloons()

        else:
            st.error(f"Error during retraining: {response.text}")

    except Exception as e:
        st.error(f"Connection error: {str(e)}")


def reset_feedback_data():
    """
    Reset all feedback data
    """
    try:
        with st.spinner("Resetting feedback data..."):
            response = requests.delete(f"{API_BASE_URL}/feedback/reset")

        if response.status_code == 200:
            st.success("✅ All feedback data has been reset!")
            st.info("You can now start collecting feedback from scratch.")
        else:
            st.error(f"Error resetting data: {response.text}")

    except Exception as e:
        st.error(f"Connection error: {str(e)}")


# Main app function
def main():
    """
    Main Streamlit app
    """
    st.set_page_config(
        page_title="CBM Expert Feedback System",
        page_icon="🔧",
        layout="wide"
    )

    # Initialize session state
    if 'api_url' not in st.session_state:
        st.session_state.api_url = API_BASE_URL

    expert_feedback_interface()


if __name__ == "__main__":
    main()