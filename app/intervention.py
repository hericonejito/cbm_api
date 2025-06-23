import os
import torch
from app.utils import data_utils
import json
from app import cbm_model

import argparse
import logging
import matplotlib.pyplot as plt
import warnings
import sys
import glob
from pathlib import Path

# In any script in app/ folder
current_file = Path(__file__)
project_root = current_file.parent.parent
shared_data_dir = project_root / "shared_data"

# Access specific folders
videos_folder = shared_data_dir / "videos_to_frame"
video_frames_folder = shared_data_dir / "video_frames"  # Updated path
# dataset_folder = shared_data_dir / "multi_train"
frames_folder = shared_data_dir / "multi_train" / "Normal"
feedback_folder = shared_data_dir / "active_learning_feedback"
predictions_folder = shared_data_dir / "predictions"  # New predictions folder
warnings.filterwarnings("always", category=FutureWarning)
warnings.simplefilter("ignore")
print = lambda *args, **kwargs: __import__('builtins').print(*args, file=sys.stderr, **kwargs)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run model inference on selected images.')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the saved model directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., cuda or cpu)')
    parser.add_argument('--images_to_display', type=int, nargs='+', default=[3,10,50], help='Indices of images to display')
    parser.add_argument('--concept_to_modify', type=str, default=None, help='Concept to modify in inference')
    parser.add_argument('--modified_value', type=float, default=0.0, help='New value of modified concept.')
    return parser.parse_args()


def load_model_and_data(model_dir, device, video_name=None, dataset_name='multi_train'):
    """
    Load model and data, with support for both video frames and dataset processing
    """
    # Load arguments from the model directory
    with open(os.path.join(model_dir, 'args.txt'), 'r') as f:
        args = json.load(f)

    dataset = args.get('dataset', dataset_name)
    _, target_preprocess = data_utils.get_target_model(args['backbone'], device)
    model = cbm_model.load_cbm(model_dir, device)

    # Load classes
    # cls_file = f"{shared_data_dir}/classes.txt"
    with open(os.path.join(shared_data_dir,'classes.txt'), 'r') as f:
        # print(cls_file)
        classes = f.read().split('\n')

    # Load concepts
    # print(model_dir)
    with open(os.path.join(model_dir, 'concepts.txt'), 'r') as f:
        concepts = f.read().split('\n')
    dataset_folder = shared_data_dir / "video_frames"/dataset_name
    # Determine data source and load accordingly
    if video_name:
        # Process specific video frames
        video_frame_path = video_frames_folder / video_name
        if video_frame_path.exists():
            print(f"Loading data from video frames: {video_frame_path}")
            val_data_t, val_pil_data = load_video_frame_data(video_frame_path, target_preprocess)
        else:
            print(f"Video frame path {video_frame_path} not found, falling back to dataset")
            val_d_probe = f"{dataset}_val"
            val_data_t = data_utils.get_data(val_d_probe, preprocess=target_preprocess, dataset_root=dataset_folder)
            val_pil_data = data_utils.get_data(val_d_probe, dataset_root=dataset_folder)
    else:
        # Use standard dataset
        val_d_probe = f"{dataset}_val"
        val_data_t = data_utils.get_data(val_d_probe, preprocess=target_preprocess, dataset_root=dataset_folder)
        val_pil_data = data_utils.get_data(val_d_probe, dataset_root=dataset_folder)

    return model, val_data_t, val_pil_data, classes, concepts, dataset


def load_video_frame_data(video_frame_path, preprocess):
    """
    Load video frame data from the video_frames directory
    """
    try:
        # Find all image files in the video frame directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(glob.glob(str(video_frame_path / ext)))
            # Also check subdirectories (like images/)
            image_files.extend(glob.glob(str(video_frame_path / "images" / ext)))

        if not image_files:
            raise ValueError(f"No image files found in {video_frame_path}")

        # Sort files to ensure consistent ordering
        image_files.sort()

        print(f"Found {len(image_files)} images in {video_frame_path}")

        # Create custom dataset loaders
        # This is a simplified version - you might need to adapt based on your data_utils structure
        val_data_t = create_tensor_dataset(image_files, preprocess)
        val_pil_data = create_pil_dataset(image_files)

        return val_data_t, val_pil_data

    except Exception as e:
        print(f"Error loading video frame data: {e}")
        raise


def create_tensor_dataset(image_files, preprocess):
    """
    Create a tensor dataset from image files
    """
    from PIL import Image
    import torch

    tensors = []
    for img_path in image_files:
        try:
            image = Image.open(img_path).convert('RGB')
            tensor = preprocess(image)
            tensors.append((tensor, 0))  # 0 as placeholder label
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    return tensors


def create_pil_dataset(image_files):
    """
    Create a PIL dataset from image files
    """
    from PIL import Image

    class PILDataset:
        def __init__(self, image_files):
            self.imgs = [(img_path, 0) for img_path in image_files]  # (path, label) format
            self.image_files = image_files

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            image = Image.open(img_path).convert('RGB')
            return image, 0  # 0 as placeholder label

    return PILDataset(image_files)
def predict_and_visualize(model, val_data_t, val_pil_data, classes, concepts, dataset, device, images_to_display):
    """
    Predict and visualize images, organizing output files by prediction class.

    Args:
        model: The trained CBM model
        val_data_t: Validation data tensor
        val_pil_data: PIL image data
        classes: List of class names
        concepts: List of concept names
        dataset: Dataset name
        device: Device to run inference on
        images_to_display: Indices of images to process

    Returns:
        predictions: List of prediction dictionaries with organized file paths
    """
    # Construct the base folder path
    base_folder = os.path.join("shared_data", dataset)

    # Create base output folder
    output_folder = dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f'Created output folder: {output_folder}')
    else:
        print('Output folder exists')

    # Create class-specific subfolders
    class_folders = {}
    for class_name in classes:
        class_folder = os.path.join(output_folder, class_name.lower())
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
            print(f'Created class folder: {class_folder}')
        class_folders[class_name] = class_folder

    predictions = []

    with torch.no_grad():
        for i in images_to_display:
            image, label = val_pil_data[i]

            # Get model prediction first to determine the folder
            x, *_ = val_data_t[i]
            x = x.unsqueeze(0).to(device)
            outputs, concept_act = model(x)
            top_logit_vals, top_classes = torch.topk(outputs[0], dim=0, k=2)
            conf = torch.nn.functional.softmax(outputs[0], dim=0)

            # Get the predicted class name
            predicted_class = classes[top_classes[0]]

            # Determine the target folder based on prediction
            target_folder = class_folders[predicted_class]

            # Create filename with original image info
            original_filename = val_pil_data.imgs[i][0].split('/')[-1]
            original_subfolder = val_pil_data.imgs[i][0].split('/')[-2]
            filename = os.path.join(target_folder, f"{original_subfolder}_{original_filename}.png")

            # Save the image to the appropriate class folder
            plt.figure(figsize=(6, 6))
            plt.imshow(image.resize([320, 320]))
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
            plt.close()  # Use plt.close() instead of plt.clf() to free memory

            # Process concept contributions
            for k in range(1):  # Only process top prediction
                contributions = concept_act[0] * model.final.weight[top_classes[k], :]
                feature_names = [("NOT " if concept_act[0][i] < 0 else "") + concepts[i]
                                 for i in range(len(concepts))]
                values = contributions.cpu().numpy()

                # Only add non-normal predictions to the results
                # if classes[top_classes[k]] != "Normal":
                predictions.append({
                    "filename": filename,
                    "class": classes[top_classes[k]],
                    "features": feature_names,
                    "values": values,
                    "confidence": f'{conf[top_classes[k]]:.3f}',
                    "original_index": i,
                    "ground_truth": classes[int(label)]
                })

    # Print summary of organized files
    print("\nFile organization summary:")
    for class_name, folder in class_folders.items():
        file_count = len([f for f in os.listdir(folder) if f.endswith('.png')])
        print(f"  {class_name}: {file_count} files in {folder}")

    return predictions


def modify_and_visualize_concept(model, val_data_t, val_pil_data, classes, concepts, dataset, device, concept_to_modify, images_to_display):
    # Construct the base folder path
    base_folder = os.path.join(os.getcwd(), "shared_data", dataset)

    img_locations = []
    with torch.no_grad():
        for i in images_to_display:
            image, label = val_pil_data[i]
            
            filename = f"{base_folder}/image.png"
            plt.imshow(image.resize([320, 320]))
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
            plt.clf()
            x, _ = val_data_t[i]
            x = x.unsqueeze(0).to(device)

            img_locations.append(filename)

            _, concept_act = model(x)
            if concept_to_modify in concepts:
                concept_act[0, concepts.index(concept_to_modify)] = 0
            outputs = model.final(concept_act)

            top_logit_vals, top_classes = torch.topk(outputs[0], dim=0, k=2)
            conf = torch.nn.functional.softmax(outputs[0], dim=0)

            for k in range(2):
                contributions = concept_act[0] * model.final.weight[top_classes[k], :]
                feature_names = [("NOT " if concept_act[0][i] < 0 else "") + concepts[i] for i in range(len(concepts))]
                values = contributions.cpu().numpy()
                max_display = min(int(sum(abs(values) > 0.005)) + 1, 8)
                title = f"Pred {k + 1}: {classes[top_classes[k]]} - Conf: {conf[top_classes[k]]:.3f} - Logit: {top_logit_vals[k]:.2f} - Bias: {model.final.bias[top_classes[k]]:.2f}"
                filename = f"{base_folder}/plot_img_{i}_prediction_{k+1}.png"
                # plots.bar(values, feature_names, max_display=max_display, title=title, fontsize=16, show=False,filename=filename)
                img_locations.append(filename)
                plt.clf()
    return img_locations


def main():
    setup_logging()
    sys_args = parse_arguments()

    model, val_data_t, val_pil_data, classes, concepts, dataset = load_model_and_data(sys_args.model_dir, sys_args.device)
    # Define paths for the output files
    output_file_path = "/tmp/intervention_output.json"
    feature_names_file_path = "/tmp/feature_names_output.json"

    if sys_args.concept_to_modify == None:
        image_locs, feature_names_list = predict_and_visualize(model, val_data_t, val_pil_data, classes, concepts, dataset, sys_args.device, sys_args.images_to_display)

        # Write image locations to the JSON file
        try:
            print("Attempting to write image locations to JSON file...")
            with open(output_file_path, "w") as f:
                json.dump(image_locs, f)
                print("Successfully wrote image locations to JSON.")
        except Exception as e:
            print(f"Failed to write image locations to file: {e}")

        # Write feature names to the JSON file
        try:
            print("Attempting to write feature names to JSON file...")
            with open(feature_names_file_path, "w") as f:
                json.dump(feature_names_list, f)
                print("Successfully wrote feature names to JSON.")
        except Exception as e:
            print(f"Failed to write feature names to file: {e}")
    else:
        image_locs = modify_and_visualize_concept(model, val_data_t, val_pil_data, classes, concepts, dataset, sys_args.device, sys_args.concept_to_modify, sys_args.images_to_display)

        # Write image locations to the JSON file
        try:
            print("Attempting to write image locations to JSON file...")
            with open(output_file_path, "w") as f:
                json.dump(image_locs, f)
                print("Successfully wrote image locations to JSON.")
        except Exception as e:
            print(f"Failed to write image locations to file: {e}")


if __name__ == "__main__":
    main()
