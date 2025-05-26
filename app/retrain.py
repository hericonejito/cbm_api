import os
import json
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image

from app.glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from app.utils import data_utils
from app.utils import model_utils
from app.utils import similarity


def retrain_cbm(
    feedback_dir: str,
    model_dir: str,
    output_dir: str,
    clip_name="ViT-B/16",
    backbone="resnet18_cub",
    feature_layer="second_to_last",
    lam=0.0007,
    proj_steps=1000,
    n_iters=1000,
    interpretability_cutoff=0.45,
    clip_cutoff=0.25,
    saga_batch_size=256,
    proj_batch_size=50000,
    batch_size=512,
    device="cpu"
):
    """
    Retrain the CBM using expert feedback stored on disk.

    Args:
        feedback_dir (str): Directory containing JSON feedback files.
        model_dir (str): Path to the original trained model (contains W_c.pt, W_g.pt, etc.).
        output_dir (str): Where to save the new retrained model.
        clip_name (str): CLIP model name.
        backbone (str): Backbone CNN model.
        feature_layer (str): Layer to extract features from.
        lam (float): Regularization parameter for GLM.
        proj_steps (int): Projection layer training steps.
        n_iters (int): Number of iterations for GLM-SAGA.
        interpretability_cutoff (float): Cutoff for concept interpretability.
        clip_cutoff (float): Cutoff for CLIP concept activation filtering.
        saga_batch_size (int): Batch size for GLM training.
        proj_batch_size (int): Batch size for projection layer training.
        batch_size (int): Batch size for activation extraction.
        device (str): Device to use ("cpu" or "cuda").
    """

    # 1. Load feedback data
    feedback = []
    for f in os.listdir(feedback_dir):
        if f.endswith(".json"):
            with open(os.path.join(feedback_dir, f), "r") as fp:
                feedback.append(json.load(fp))

    if len(feedback) == 0:
        raise ValueError("No feedback data found.")

    print(f"Loaded {len(feedback)} feedback samples.")

    # 2. Preprocess and extract activations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    images = []
    labels = []

    for entry in feedback:
        img_path = entry["image_path"]
        label = entry["true_label"]

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        images.append(img_tensor)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)

    # 3. Extract features and CLIP projections
    activations_dir = "retrain_activations"
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)

    model_utils.save_activations(
        clip_name=clip_name,
        target_name=backbone,
        target_layers=[feature_layer],
        d_probe=None,
        dataset_root=None,
        images=images,
        labels=labels.tolist(),
        concept_set=None,
        batch_size=batch_size,
        device=device,
        pool_mode="avg",
        save_dir=activations_dir,
        mode="retrain"
    )

    target_feats_path = os.path.join(activations_dir, "target_features.pt")
    clip_feats_path = os.path.join(activations_dir, "clip_features.pt")
    text_feats_path = os.path.join(activations_dir, "text_features.pt")

    with torch.no_grad():
        target_features = torch.load(target_feats_path, map_location=device).float()
        image_features = torch.load(clip_feats_path, map_location=device).float()
        text_features = torch.load(text_feats_path, map_location=device).float()

        image_features /= torch.norm(image_features, dim=1, keepdim=True)
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        clip_features = image_features @ text_features.T

    # 4. Filter concepts
    topk_mean = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)
    keep_indices = topk_mean > clip_cutoff
    clip_features = clip_features[:, keep_indices]
    text_features = text_features[keep_indices]

    # 5. Train projection layer
    proj_layer = torch.nn.Linear(target_features.shape[1], text_features.shape[0], bias=False).to(device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)

    best_loss = float("inf")
    best_weights = None
    indices = list(range(len(target_features)))
    for i in range(proj_steps):
        batch_idx = random.sample(indices, min(proj_batch_size, len(indices)))
        batch = torch.LongTensor(batch_idx)
        outs = proj_layer(target_features[batch].to(device))
        loss = -similarity.cos_similarity_cubed_single(clip_features[batch].to(device), outs).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

        if i % 100 == 0 or i == proj_steps - 1:
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_weights = proj_layer.weight.detach().clone()

    proj_layer.load_state_dict({"weight": best_weights})

    # 6. Normalize concepts and filter by interpretability
    with torch.no_grad():
        outs = proj_layer(target_features.to(device))
        sim = similarity.cos_similarity_cubed_single(clip_features.to(device), outs)
        keep = sim > interpretability_cutoff

    W_c = proj_layer.weight[keep]
    proj_layer = torch.nn.Linear(target_features.shape[1], W_c.shape[0], bias=False)
    proj_layer.load_state_dict({"weight": W_c})

    # 7. Project to concept space and standardize
    with torch.no_grad():
        concepts = proj_layer(target_features)
        mean = concepts.mean(0, keepdim=True)
        std = concepts.std(0, keepdim=True)
        concepts = (concepts - mean) / std

    indexed_ds = IndexedTensorDataset(concepts, labels)
    loader = DataLoader(indexed_ds, batch_size=saga_batch_size, shuffle=True)

    # 8. GLM training
    linear = torch.nn.Linear(concepts.shape[1], labels.max().item() + 1).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    metadata = {'max_reg': {'nongrouped': lam}}
    output_proj = glm_saga(
        linear, loader,
        STEP_SIZE=0.1, N_ITER=n_iters,
        ALPHA=0.99, epsilon=1, k=1,
        val_loader=None, do_zero=False,
        metadata=metadata, n_ex=len(concepts), n_classes=labels.max().item() + 1
    )

    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']

    # 9. Save updated model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(mean, os.path.join(output_dir, "proj_mean.pt"))
    torch.save(std, os.path.join(output_dir, "proj_std.pt"))
    torch.save(W_c, os.path.join(output_dir, "W_c.pt"))
    torch.save(W_g, os.path.join(output_dir, "W_g.pt"))
    torch.save(b_g, os.path.join(output_dir, "b_g.pt"))

    print(f"Retrained CBM saved to: {output_dir}")
