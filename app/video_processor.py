import cv2
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import logging
import argparse
import matplotlib.pyplot as plt
# parser = argparse.ArgumentParser(description='Concept Extraction Script')
# parser.add_argument('--video_name', type=str, required=False, help='Video name from Streamlit')
# args = parser.parse_args()
#
# logging.info(f"Script started with arguments: {args}")

# Construct the base folder path
# base_folder = os.path.join(os.getcwd(), "shared_data_1")
def extract_frames(video_path, output_folder, frame_interval=10):
    """Extracts frames from a video file."""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame_number = True, 0

    frames = []

    while success:
        success, frame = cap.read()
        if frame_number % frame_interval == 0 and success:
            frame_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append((frame_path, frame))
        frame_number += 1

    cap.release()
    return frames

def grad_cam(model, target_layer, frame_tensor):
    """Computes Grad-CAM heatmap for a given frame tensor."""
    gradients = []
    activations = []

    def save_gradients(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def save_activations(module, input, output):
        activations.append(output)

    # Register hooks
    handle_grad = target_layer.register_backward_hook(save_gradients)
    handle_act = target_layer.register_forward_hook(save_activations)

    # Forward pass
    output = model(frame_tensor)
    output_idx = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    output[0, output_idx].backward()

    # Remove hooks
    handle_grad.remove()
    handle_act.remove()

    # Compute Grad-CAM
    grad = gradients[0].cpu().numpy()
    act = activations[0].detach().cpu().numpy()

    grad = np.mean(grad, axis=(2, 3), keepdims=True)
    cam = np.sum(grad * act, axis=1).squeeze()
    cam = np.maximum(cam, 0)  # ReLU to keep only positive values
    cam = cam / cam.max()  # Normalize to [0, 1]

    return cam

def extract_embeddings(frames, model, preprocess, target_layer, device):
    """Extracts embeddings and Grad-CAM heatmaps for each frame."""
    embeddings = []
    heatmaps = []

    for _, frame in frames:
        frame_tensor = preprocess(frame).unsqueeze(0).to(device)

        # Extract embeddings
        with torch.no_grad():
            outputs = model(frame_tensor)
            embedding = outputs.cpu().detach().numpy().flatten()
            embeddings.append(embedding)

        # Compute Grad-CAM heatmap
        cam = grad_cam(model, target_layer, frame_tensor)
        heatmaps.append(cam)

    return np.array(embeddings), heatmaps

def visualize_heatmap(frame, heatmap, output_path, label):
    """Overlays the heatmap on the original frame and saves the visualization."""
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap_normalized = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
    cv2.putText(overlay, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(output_path, overlay)

def classify_outliers(embeddings):
    """Classifies frames as outliers or not using Isolation Forest."""
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    model = IsolationForest(contamination=0.1, random_state=42)
    predictions = model.fit_predict(scaled_embeddings)
    return predictions  # -1 indicates an outlier

def label_outlier_types(outlier_embeddings, num_clusters=3):
    """Cluster outlier embeddings and assign labels to clusters."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(outlier_embeddings)

    # Map cluster labels to anomaly types (manual mapping based on inspection)
    anomaly_types = {0: "Corrosion", 1: "Leakage", 2: "Crack"}
    labeled_clusters = [anomaly_types.get(label, "Unknown") for label in cluster_labels]

    return labeled_clusters

def process_video(video_path, output_folder,video_name, frame_interval=10):
    """Processes a video and classifies frames as outliers or not."""
    print("Extracting frames...")
    frames = extract_frames(video_path, output_folder, frame_interval)

    # print("Loading model...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # base_model = resnet50(pretrained=True)
    # base_model.fc = nn.Identity()  # Remove the classification head
    # target_layer = base_model.layer4[2].conv3  # Specify target layer for Grad-CAM
    # base_model = base_model.to(device)
    # base_model.eval()
    #
    # preprocess = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    #
    # print("Extracting embeddings and Grad-CAM heatmaps...")
    # embeddings, heatmaps = extract_embeddings(frames, base_model, preprocess, target_layer, device)
    #
    # print("Classifying frames...")
    # predictions = classify_outliers(embeddings)
    #
    # # Handle outlier labeling
    # outlier_indices = np.where(predictions == -1)[0]
    # outlier_embeddings = embeddings[outlier_indices]
    # outlier_labels = label_outlier_types(outlier_embeddings)
    #
    results = []
    # outlier_counter = 0
    for frame_path, frame in frames:
        dataset_folder = os.path.join(output_folder, "images")
        # outlier_folder = os.path.join(outlier_folder, outlier_labels[outlier_counter])
        os.makedirs(dataset_folder, exist_ok=True)

        frame_save_path = os.path.join(dataset_folder, f"{video_name}_{frame}.jpg")
        # heatmap_save_path = os.path.join(outlier_folder, f"{video_name}_frame_{i}_heatmap.jpg")

        cv2.imwrite(frame_save_path, frame)
    # for i, ((frame_path, frame), prediction, heatmap) in enumerate(zip(frames, predictions, heatmaps)):
    #     if prediction == -1:
    #         for z in ['train','test']:
    #             label = f"Outlier ({outlier_labels[outlier_counter]})"
    #             outlier_folder = os.path.join(output_folder, z)
    #             outlier_folder = os.path.join(outlier_folder, outlier_labels[outlier_counter])
    #             os.makedirs(outlier_folder, exist_ok=True)
    #
    #             frame_save_path = os.path.join(outlier_folder, f"{video_name}_frame_{i}.jpg")
    #             # heatmap_save_path = os.path.join(outlier_folder, f"{video_name}_frame_{i}_heatmap.jpg")
    #
    #             cv2.imwrite(frame_save_path, frame)
    #             # visualize_heatmap(frame, heatmap, heatmap_save_path, label)
    #
    #         outlier_counter += 1
    #     else:
    #         for z in ['train','test']:
    #             label = "Normal"
    #             normal_folder = os.path .join(output_folder, z)
    #             normal_folder = os.path .join(normal_folder, "Normal")
    #             os.makedirs(normal_folder, exist_ok=True)
    #
    #             frame_save_path = os.path.join(normal_folder, f"{video_name}_frame_{i}.jpg")
    #             # heatmap_save_path = os.path.join(normal_folder, f"{video_name}_frame_{i}_heatmap.jpg")
    #
    #             cv2.imwrite(frame_save_path, frame)
    #             # visualize_heatmap(frame, heatmap, heatmap_save_path, label)

        results.append(frame_path)
        print(f"Frame: {frame_path}")

    return results

if __name__ == "__main__":
    for video_name in ['input_video','input_file_5044','input_file_209']: #The video file names that we want to be framed
        video_path = f"videos_to_frame/{video_name}.mp4"  # Replace with your video path
        output_folder = f"shared_data/multi_train/Normal"

        results = process_video(video_path, output_folder,video_name, frame_interval=30)
        print("Processing complete.")
