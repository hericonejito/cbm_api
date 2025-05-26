import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os
import json
FASTAPI_URL = "http://localhost:8000"

st.title("CBM Video Outlier Detector")
# Session state to track processing
if "video_uploaded" not in st.session_state:
    st.session_state.video_uploaded = False
if "video_name" not in st.session_state:
    st.session_state.video_name = None
if "detection_done" not in st.session_state:
    st.session_state.detection_done = False
st.header('Step 1: Upload and frame your video')
# Upload video
video_file = st.file_uploader("Upload your video", type=["mp4"])

if video_file is not None:
    st.video(video_file)

    if st.button("Process Video"):
        with st.spinner("Uploading and processing video..."):
            files = {"file": (video_file.name, video_file, "video/mp4")}
            response = requests.post(f"{FASTAPI_URL}/upload_video/", files=files)

        if response.status_code == 200:
            results = response.json()["results"]
            st.success("Framing complete.")
            st.session_state.video_uploaded = True
            st.session_state.video_name = os.path.splitext(video_file.name)[0]

            for result in results:
                st.image(f"{FASTAPI_URL}/{result['frame_path']}", caption=f"{result['frame_path']}")

        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
# Step 2: Trigger outlier detection after framing
if st.session_state.video_uploaded and not st.session_state.detection_done:
    st.header("Step 2: Run Outlier Detection")
    if st.button("Detect Anomalies"):
        with st.spinner("Running model..."):
            payload = {"video_name": st.session_state.video_name}
            response = requests.post(f"{FASTAPI_URL}/predict_frames/", json=payload)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                st.success("Outlier detection complete. See results below.")
                st.session_state.detection_done = True

                st.header("Detected Outlier Frames")
                results = json.loads(results)
                for item in results:
                    st.image(FASTAPI_URL + "/" + item["path"], caption=f"{item['label']} — {item['confidence']} - {item['frame_id']}")
            else:
                st.info("No outliers were detected.")
        else:
            st.error("Detection failed.")