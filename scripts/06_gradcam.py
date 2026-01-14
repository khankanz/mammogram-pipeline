#!/usr/bin/env python3
"""GradCAM visualization for EfficientNet-B0 model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b0
import torchvision.transforms as T

from lib.config import MODEL_DIR, DB_PATH
from lib.db import get_db, get_labeled_images


class GradCAM:
    """GradCAM implementation for visualizing CNN attention."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class):
        """Generate GradCAM heatmap for target class."""
        self.model.eval()

        output = self.model(input_tensor)
        self.model.zero_grad()

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()


def create_model(num_classes=2, dropout=0.4):
    """Create EfficientNet-B0 for multi-label classification."""
    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, num_classes)
    )
    return model


def load_model(model_path: Path, device: torch.device):
    """Load trained model."""
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path, device: torch.device):
    """Load and preprocess image."""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img.resize((224, 224)))
    img_tensor = transform(img).unsqueeze(0).to(device)

    return img_tensor, img_array


def generate_gradcam_visualization(
    model_path: Path,
    output_dir: Path,
    db_path: Path,
    num_samples: int = 16
):
    """Generate GradCAM visualizations for sample images."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(model_path, device)

    # Target layer: last conv layer in EfficientNet-B0 features
    target_layer = model.features[-1]

    gradcam = GradCAM(model, target_layer)

    db = get_db(db_path)
    labeled = get_labeled_images(db)

    # Sample from each category
    biopsy_only = [r for r in labeled if r['has_biopsy_tool'] == 1 and r['has_mag_view'] == 0]
    mag_only = [r for r in labeled if r['has_biopsy_tool'] == 0 and r['has_mag_view'] == 1]
    both = [r for r in labeled if r['has_biopsy_tool'] == 1 and r['has_mag_view'] == 1]
    neither = [r for r in labeled if r['has_biopsy_tool'] == 0 and r['has_mag_view'] == 0]

    samples = []
    np.random.seed(42)
    for category, name in [(biopsy_only, 'biopsy'), (mag_only, 'magview'),
                           (both, 'both'), (neither, 'neither')]:
        if category:
            n = min(num_samples // 4, len(category))
            indices = np.random.choice(len(category), n, replace=False)
            for i in indices:
                samples.append((category[i], name))

    print(f"Generating GradCAM for {len(samples)} images...")

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (row, category) in enumerate(samples):
        thumb_path = Path(row['thumbnail_path'])
        if not thumb_path.exists():
            continue

        img_tensor, img_array = preprocess_image(thumb_path, device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        cam_biopsy = gradcam.generate(img_tensor, target_class=0)
        cam_mag = gradcam.generate(img_tensor, target_class=1)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(img_array)
        axes[0].set_title(f'Original\nGT: biopsy={int(row["has_biopsy_tool"])}, mag={int(row["has_mag_view"])}')
        axes[0].axis('off')

        axes[1].imshow(img_array)
        axes[1].imshow(cam_biopsy, cmap='jet', alpha=0.5)
        axes[1].set_title(f'Biopsy Tool Focus\nPred: {probs[0]:.2f}')
        axes[1].axis('off')

        axes[2].imshow(img_array)
        axes[2].imshow(cam_mag, cmap='jet', alpha=0.5)
        axes[2].set_title(f'Mag View Focus\nPred: {probs[1]:.2f}')
        axes[2].axis('off')

        combined = (cam_biopsy + cam_mag) / 2
        axes[3].imshow(img_array)
        axes[3].imshow(combined, cmap='jet', alpha=0.5)
        axes[3].set_title('Combined Focus')
        axes[3].axis('off')

        plt.tight_layout()

        output_path = output_dir / f'gradcam_{category}_{idx:03d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")

    print(f"\nGradCAM visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate GradCAM visualizations")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "model.pth")
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "gradcam")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--image", type=Path, help="Single image to visualize")

    args = parser.parse_args()

    if not args.model.exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    generate_gradcam_visualization(args.model, args.output_dir, args.db, args.num_samples)


if __name__ == "__main__":
    main()
