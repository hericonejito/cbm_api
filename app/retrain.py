import os
import json
import torch
import shutil
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from app.glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from app.utils import data_utils


def retrain_cbm(
    feedback_dir: str,
    model_dir: str,
    output_dir: str = None,
    device: str = "cpu",
    lam: float = 0.0007,
    n_iters: int = 1000,
    saga_batch_size: int = 256,
):
    """
    Fine-tune the CBM final layer (W_g) using expert feedback.

    This function keeps the projection layer (W_c) and normalization parameters
    frozen, only retraining the final classification layer using GLM-SAGA.
    This is appropriate for active learning with limited feedback samples.

    Args:
        feedback_dir: Directory containing annotations.json and class subfolders with images.
        model_dir: Path to the trained model directory (contains W_c.pt, W_g.pt, etc.).
        output_dir: Where to save the retrained model. If None, updates model_dir in place.
        device: Device to use ("cpu" or "cuda").
        lam: Sparsity regularization parameter for GLM-SAGA.
        n_iters: Number of iterations for GLM-SAGA.
        saga_batch_size: Batch size for GLM-SAGA training.

    Returns:
        dict: Training results including metrics and paths.
    """

    # Use model_dir as output if not specified (in-place update)
    if output_dir is None:
        output_dir = model_dir

    # 1. Load model configuration and existing weights
    print(f"Loading model from: {model_dir}")

    args_path = os.path.join(model_dir, "args.txt")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Model args not found at {args_path}")

    with open(args_path, 'r') as f:
        model_args = json.load(f)

    backbone_name = model_args.get('backbone', 'resnet18_cub')

    # Load existing model components (these stay frozen)
    W_c = torch.load(os.path.join(model_dir, "W_c.pt"), map_location=device)
    proj_mean = torch.load(os.path.join(model_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(model_dir, "proj_std.pt"), map_location=device)

    # Load old W_g for backup
    W_g_old = torch.load(os.path.join(model_dir, "W_g.pt"), map_location=device)
    b_g_old = torch.load(os.path.join(model_dir, "b_g.pt"), map_location=device)

    print(f"Loaded model with backbone: {backbone_name}")
    print(f"Projection layer shape: {W_c.shape} (frozen)")
    print(f"Final layer shape: {W_g_old.shape} (will be retrained)")

    # 2. Load class mapping
    classes_file = os.path.join(os.path.dirname(feedback_dir), "classes.txt")
    if not os.path.exists(classes_file):
        # Fallback to default classes
        classes = ["Corrosion", "Crack", "Leakage", "Normal"]
        print(f"Using default classes: {classes}")
    else:
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded classes from file: {classes}")

    # Create lowercase mapping for matching annotations
    class_to_idx = {cls.lower(): idx for idx, cls in enumerate(classes)}
    n_classes = len(classes)

    # 3. Load feedback annotations
    annotations_path = os.path.join(feedback_dir, "annotations.json")
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations not found at {annotations_path}")

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    if len(annotations) == 0:
        raise ValueError("No feedback annotations found.")

    print(f"Loaded {len(annotations)} feedback samples")

    # 4. Load backbone model for feature extraction
    print(f"Loading backbone: {backbone_name}")
    backbone_model, preprocess = data_utils.get_target_model(backbone_name, device)
    backbone_model.eval()

    # For resnet18_cub, use features extractor (excluding final FC)
    if "cub" in backbone_name:
        feature_extractor = lambda x: backbone_model.features(x)
    else:
        feature_extractor = torch.nn.Sequential(*list(backbone_model.children())[:-1])

    # 5. Create projection layer from frozen W_c
    proj_layer = torch.nn.Linear(W_c.shape[1], W_c.shape[0], bias=False).to(device)
    proj_layer.load_state_dict({"weight": W_c})
    proj_layer.eval()

    # 6. Process feedback images and extract concept activations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    concept_activations = []
    labels = []
    skipped = 0

    print("Extracting features from feedback images...")
    with torch.no_grad():
        for annotation in tqdm(annotations):
            # Get image path - handle both Docker and local paths
            img_path = annotation.get("image_path", "")

            # Try alternative path if original doesn't exist
            if not os.path.exists(img_path):
                # Try replacing /app/ with actual path
                alt_path = img_path.replace("/app/", "")
                if os.path.exists(alt_path):
                    img_path = alt_path
                else:
                    # Try constructing from feedback_dir
                    expert_class = annotation.get("expert_classification", "").lower()
                    feedback_id = annotation.get("feedback_id", "")
                    frame_id = annotation.get("frame_id", "")
                    possible_path = os.path.join(feedback_dir, expert_class, f"{feedback_id}_{frame_id}.jpg")
                    if os.path.exists(possible_path):
                        img_path = possible_path
                    else:
                        print(f"Warning: Image not found: {img_path}")
                        skipped += 1
                        continue

            # Get label
            expert_class = annotation.get("expert_classification", "").lower()
            if expert_class not in class_to_idx:
                print(f"Warning: Unknown class '{expert_class}', skipping")
                skipped += 1
                continue

            label_idx = class_to_idx[expert_class]

            # Load and preprocess image
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
                skipped += 1
                continue

            # Extract backbone features
            features = feature_extractor(img_tensor)
            features = torch.flatten(features, 1)

            # Project to concept space
            concepts = proj_layer(features)

            # Normalize with frozen mean/std
            concepts_norm = (concepts - proj_mean) / proj_std

            concept_activations.append(concepts_norm.cpu())
            labels.append(label_idx)

    if len(concept_activations) == 0:
        raise ValueError(f"No valid feedback samples found. Skipped {skipped} samples.")

    print(f"Successfully processed {len(concept_activations)} samples ({skipped} skipped)")

    # Stack tensors
    X = torch.cat(concept_activations, dim=0)
    y = torch.tensor(labels, dtype=torch.long)

    print(f"Training data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {torch.bincount(y).tolist()}")

    # 7. Train new final layer using GLM-SAGA
    print(f"\nTraining final layer with GLM-SAGA (lam={lam}, n_iters={n_iters})...")

    # Create indexed dataset and loader
    indexed_ds = IndexedTensorDataset(X, y)
    train_loader = DataLoader(indexed_ds, batch_size=saga_batch_size, shuffle=True)

    # Initialize new final layer (zero-initialized as per original training)
    n_concepts = X.shape[1]
    linear = torch.nn.Linear(n_concepts, n_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    # GLM-SAGA parameters (matching original training)
    STEP_SIZE = 0.1
    ALPHA = 0.99

    metadata = {'max_reg': {'nongrouped': lam}}

    output_proj = glm_saga(
        linear,
        train_loader,
        STEP_SIZE,
        n_iters,
        ALPHA,
        epsilon=1,
        k=1,
        val_loader=None,
        do_zero=False,
        metadata=metadata,
        n_ex=len(X),
        n_classes=n_classes
    )

    # Extract trained weights
    W_g_new = output_proj['path'][0]['weight']
    b_g_new = output_proj['path'][0]['bias']

    # 8. Save updated model
    os.makedirs(output_dir, exist_ok=True)

    # Backup old weights if updating in place
    if output_dir == model_dir:
        backup_dir = os.path.join(model_dir, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy(os.path.join(model_dir, "W_g.pt"), os.path.join(backup_dir, "W_g.pt"))
        shutil.copy(os.path.join(model_dir, "b_g.pt"), os.path.join(backup_dir, "b_g.pt"))
        print(f"Backed up old weights to: {backup_dir}")

    # Save new final layer weights
    torch.save(W_g_new, os.path.join(output_dir, "W_g.pt"))
    torch.save(b_g_new, os.path.join(output_dir, "b_g.pt"))

    # Copy frozen components if saving to new directory
    if output_dir != model_dir:
        torch.save(W_c, os.path.join(output_dir, "W_c.pt"))
        torch.save(proj_mean, os.path.join(output_dir, "proj_mean.pt"))
        torch.save(proj_std, os.path.join(output_dir, "proj_std.pt"))

        # Copy other files
        for fname in ["args.txt", "concepts.txt"]:
            src = os.path.join(model_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(output_dir, fname))

    # Save retraining metrics
    metrics = {
        "retrain_timestamp": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_classes": n_classes,
        "class_distribution": torch.bincount(y).tolist(),
        "lam": float(output_proj['path'][0]['lam']),
        "lr": float(output_proj['path'][0]['lr']),
        "alpha": float(output_proj['path'][0]['alpha']),
        "time": float(output_proj['path'][0]['time']),
        "metrics": output_proj['path'][0]['metrics'],
        "sparsity": {
            "non_zero_weights": int((W_g_new.abs() > 1e-5).sum().item()),
            "total_weights": int(W_g_new.numel()),
            "percentage_non_zero": float((W_g_new.abs() > 1e-5).sum().item() / W_g_new.numel())
        }
    }

    metrics_path = os.path.join(output_dir, "retrain_metrics.txt")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nRetraining complete!")
    print(f"  Samples used: {len(X)}")
    print(f"  Training accuracy: {metrics['metrics']['acc_tr']:.4f}")
    print(f"  Sparsity: {metrics['sparsity']['percentage_non_zero']:.2%} non-zero weights")
    print(f"  Model saved to: {output_dir}")

    return {
        "status": "success",
        "output_dir": output_dir,
        "metrics": metrics
    }
