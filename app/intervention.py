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

def load_model_and_data(model_dir, device):
    # Load arguments from the model directory
    with open(os.path.join(model_dir, 'args.txt'), 'r') as f:
        args = json.load(f)
    dataset = args['dataset']
    _, target_preprocess = data_utils.get_target_model(args['backbone'], device)
    model = cbm_model.load_cbm(model_dir, device)

    val_d_probe = f"{dataset}_val"
    # get concept set
    cls_file = f"shared_data/classes.txt"

    val_data_t = data_utils.get_data(val_d_probe, preprocess=target_preprocess,dataset_root=f"shared_data/{dataset}")
    val_pil_data = data_utils.get_data(val_d_probe,dataset_root=f"shared_data/{dataset}")

    with open(cls_file, 'r') as f:
        classes = f.read().split('\n')

    with open(os.path.join(model_dir, 'concepts.txt'), 'r') as f:
        concepts = f.read().split('\n')

    return model, val_data_t, val_pil_data, classes, concepts, dataset


def predict_and_visualize(model, val_data_t, val_pil_data, classes, concepts, dataset, device, images_to_display):
    # Construct the base folder path
    base_folder = os.path.join( "shared_data", dataset)
    if os.path.exists(f'{base_folder}/output'):
        print('Folder Output exists')
    else:
        os.makedirs(f'{base_folder}/output')
    base_folder = f'{base_folder}/output'
    predictions = []
    with torch.no_grad():
        for i in images_to_display:
            image, label = val_pil_data[i]
            
            filename = f"{base_folder}/{val_pil_data.imgs[i][0].split('/')[-2]}_{val_pil_data.imgs[i][0].split('/')[-1]}.png"
            plt.imshow(image.resize([320, 320]))
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
            plt.clf()
            x, _ = val_data_t[i]
            x = x.unsqueeze(0).to(device)

            # img_locations.append(filename)

            outputs, concept_act = model(x)
            top_logit_vals, top_classes = torch.topk(outputs[0], dim=0, k=2)
            conf = torch.nn.functional.softmax(outputs[0], dim=0)
            # print(f"Image: {i} | Gt: {classes[int(label)]} | 1st Pred: {classes[top_classes[0]]}, {top_logit_vals[0]:.3f} | 2nd Pred: {classes[top_classes[1]]}, {top_logit_vals[1]:.3f}")

            for k in range(1):
                contributions = concept_act[0] * model.final.weight[top_classes[k], :]
                feature_names = [("NOT " if concept_act[0][i] < 0 else "") + concepts[i] for i in range(len(concepts))]
                values = contributions.cpu().numpy()
                # max_display = min(int(sum(abs(values) > 0.005)) + 1, 8)
                # title = f"Pred {k + 1}: {classes[top_classes[k]]} - Conf: {conf[top_classes[k]]:.3f} - Logit: {top_logit_vals[k]:.2f} - Bias: {model.final.bias[top_classes[k]]:.2f}"
                # filename = f"{base_folder}/plot_img_{i}_prediction_{k+1}.png"
                # plots.bar(values, feature_names, max_display=max_display, title=title, fontsize=16, show=False,filename=filename)
                if classes[top_classes[k]]!="Normal":
                    predictions.append({"filename":filename,"class":classes[top_classes[k]],"features":feature_names,"values":values,"confidence":f'{conf[top_classes[k]]:.3f}'})
                # feature_names_list.extend(feature_names)
                # plt.clf()

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
