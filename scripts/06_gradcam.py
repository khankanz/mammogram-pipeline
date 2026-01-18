#!/usr/bin/env python3
"""GradCAM visualization: Where is the model actually looking?

GradCAM (Gradient-weighted Class Activation Mapping) answers: "For a given prediction,
which regions of the input image most influenced that prediction?"

This is your debugging tool for "the model is confident but wrong" cases.

=============================================================================
BACKGROUND: HOW CNNs PROCESS IMAGES
=============================================================================

A CNN is a pipeline of convolutional layers. Each layer:
1. Applies filters (small sliding windows) to detect patterns
2. Often downsamples (shrinks spatial dimensions)

EfficientNet-B0's journey through your 224x224 image:

    Input:           224 × 224 × 3      (RGB image: height × width × channels)
    After layer 1:   112 × 112 × 32     (32 filters, spatial halved)
    After layer 2:   56 × 56 × 24       (more abstract patterns)
    ...
    Final conv:      7 × 7 × 1280       (high-level semantic features)

By the end, the image is a 7×7 grid with 1280 "pattern detectors" per cell.
Each of those 1280 channels answers: "how much of pattern X is here?"

The classifier head then:
    7×7×1280 → global avg pool → 1280-dim vector → linear → 2 outputs (biopsy, mag)

=============================================================================
THE GRADCAM ALGORITHM
=============================================================================

1. Forward pass: image → activations (7×7×1280) → prediction (e.g., biopsy=0.97)

2. Backward pass from target class output:
   - Compute gradients of the 7×7×1280 activations
   - Gradient = "if I increased this value, how much would the target score change?"

3. Importance weights (global average pooling on gradients):
   - For each of 1280 feature maps, average gradients spatially → 1280 scalars
   - High weight = this feature map strongly supports the prediction

4. Weighted combination:
   - Multiply each 7×7 feature map by its importance weight
   - Sum all 1280 weighted feature maps → single 7×7 heatmap

5. ReLU + resize:
   - Zero out negative values (we want "pro-target" regions only)
   - Upsample 7×7 → 224×224 to overlay on original image

The result: a heatmap showing WHERE the model looked to make its decision.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.models import efficientnet_b0

from fastcore.all import *
from fastcore.script import call_parse
from fastprogress import progress_bar

from lib.config import MODEL_DIR, DB_PATH, IMAGE_SIZE
from lib.db import get_db, get_labeled

# =============================================================================
# GRADCAM IMPLEMENTATION
# =============================================================================

class GradCAM:
    """GradCAM: Visualize which image regions drive a specific prediction.
    
    The key insight: gradients tell you sensitivity. If a feature map has a large
    positive gradient w.r.t. the output, increasing that feature's activation
    would increase the prediction confidence. That feature is RELEVANT.
    
    By weighting spatial feature maps by their gradients, we get a map of
    "where the relevant features are activated."
    """
    
    def __init__(self, model, target_layer):
        """Register hooks to capture activations and gradients.
        
        PyTorch discards intermediate values to save memory. Hooks let us intercept them.
        
        Args:
            model: The CNN model (must be in eval mode)
            target_layer: Which layer to visualize (typically last conv layer)
                         For EfficientNet-B0: model.features[-1] gives 7×7×1280 output
        """
        self.model, self.target_layer = model, target_layer
        self.gradients, self.activations = None, None
        
        # Forward hook: fires during forward pass, captures feature maps
        # These are the 7×7×1280 activations we'll weight and sum
        target_layer.register_forward_hook(self._save_activation)
        
        # Backward hook: fires during backward pass, captures gradients
        # These tell us "importance" of each feature map for the target class
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, inp, out):
        "Hook callback: save feature maps during forward pass"
        # out shape: (batch=1, channels=1280, height=7, width=7)
        self.activations = out.detach()
    
    def _save_gradient(self, module, grad_inp, grad_out):
        "Hook callback: save gradients during backward pass"
        # grad_out[0] shape: (batch=1, channels=1280, height=7, width=7)
        # This is ∂(target_score)/∂(activation) for each spatial location
        self.gradients = grad_out[0].detach()
    
    def generate(self, x, target_class):
        """Generate GradCAM heatmap for a specific class.
        
        Args:
            x: Input tensor, shape (1, 3, 224, 224)
            target_class: Which output to explain (0=biopsy, 1=mag)
        
        Returns:
            Numpy array (224, 224) with values 0-1, higher = more important
        """
        self.model.eval()
        
        # === FORWARD PASS ===
        # Image flows through network, hooks capture the 7×7×1280 activations
        out = self.model(x)  # out shape: (1, 2) for our binary classifier
        
        # === BACKWARD PASS FROM TARGET CLASS ===
        # We want gradients w.r.t. ONLY the target class score.
        # Create a "fake gradient" that's 1 for target, 0 for others.
        # This says: "compute how activations affect THIS specific output"
        self.model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, target_class] = 1
        
        # Backward pass: gradients flow from output back through network
        # Our hook captures ∂(target_score)/∂(activations) at target_layer
        out.backward(gradient=one_hot, retain_graph=True)
        
        # === COMPUTE IMPORTANCE WEIGHTS ===
        # For each of 1280 feature maps, compute ONE importance score.
        # Method: global average pooling on gradients (mean over 7×7 spatial dims)
        #
        # Why average? If a feature map's gradients are positive everywhere,
        # that feature map consistently helps the prediction across all locations.
        # If gradients are mixed +/-, the feature map's contribution is ambiguous.
        #
        # weights shape: (1, 1280, 1, 1) - one scalar per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # === WEIGHTED COMBINATION OF FEATURE MAPS ===
        # Each 7×7 feature map gets multiplied by its importance weight, then summed.
        #
        # If feature map #200 detects "bright metallic objects" and has high weight,
        # locations where #200 is activated will dominate the final heatmap.
        #
        # cam shape: (1, 1, 7, 7) - weighted sum of all 1280 feature maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # === RELU: KEEP ONLY POSITIVE INFLUENCE ===
        # Negative values mean "this region pushed AWAY from the prediction"
        # We only care about regions that SUPPORT the prediction
        cam = F.relu(cam)
        
        # === NORMALIZE TO 0-1 ===
        cam = cam - cam.min()
        if cam.max() > 0: cam = cam / cam.max()
        
        # === RESIZE TO INPUT DIMENSIONS ===
        # 7×7 → 224×224 via bilinear interpolation
        # Now we can overlay on the original image
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), 
                           mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()

# =============================================================================
# MODEL UTILITIES
# =============================================================================

def create_model(n_out=2, dropout=0.4):
    "Create EfficientNet-B0 with custom classifier head"
    model = efficientnet_b0(weights=None)
    n_in = model.classifier[1].in_features  # 1280 for EfficientNet-B0
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_in, n_out))
    return model


def load_model(model_path, device='cpu'):
    "Load trained model weights into architecture"
    model = create_model()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    return model

# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

# ImageNet normalization - match what the pretrained model expects
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def get_preprocess():
    "Standard preprocessing: resize, tensor, normalize"
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_image(path, device='cpu'):
    """Load and preprocess image, return (tensor, numpy_array).
    
    Returns both because we need:
    - tensor for model input
    - numpy array for visualization overlay
    """
    tfm = get_preprocess()
    img = Image.open(path).convert('RGB')
    arr = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))  # For display
    tensor = tfm(img).unsqueeze(0).to(device)              # For model
    return tensor, arr

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_gradcam(img_arr, cam_biopsy, cam_mag, probs, labels, save_path=None):
    """Create 4-panel GradCAM visualization.
    
    Panels: [Original] [Biopsy Focus] [Mag View Focus] [Combined]
    
    Args:
        img_arr: Original image as numpy array (224, 224, 3)
        cam_biopsy: GradCAM heatmap for biopsy class (224, 224)
        cam_mag: GradCAM heatmap for mag view class (224, 224)
        probs: Model predictions [prob_biopsy, prob_mag]
        labels: Ground truth dict with 'has_biopsy_tool', 'has_mag_view'
        save_path: If provided, save figure here
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    gt_biopsy, gt_mag = int(labels['has_biopsy_tool']), int(labels['has_mag_view'])
    
    # Panel 1: Original image with ground truth
    axes[0].imshow(img_arr)
    axes[0].set_title(f'Original\nGT: biopsy={gt_biopsy}, mag={gt_mag}')
    axes[0].axis('off')
    
    # Panel 2: Biopsy focus heatmap
    # Overlay shows WHERE the model looked to decide "biopsy tool present"
    axes[1].imshow(img_arr)
    axes[1].imshow(cam_biopsy, cmap='jet', alpha=0.5)
    axes[1].set_title(f'Biopsy Tool Focus\nPred: {probs[0]:.2f}')
    axes[1].axis('off')
    
    # Panel 3: Mag view focus heatmap
    axes[2].imshow(img_arr)
    axes[2].imshow(cam_mag, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Mag View Focus\nPred: {probs[1]:.2f}')
    axes[2].axis('off')
    
    # Panel 4: Combined focus (average of both heatmaps)
    combined = (cam_biopsy + cam_mag) / 2
    axes[3].imshow(img_arr)
    axes[3].imshow(combined, cmap='jet', alpha=0.5)
    axes[3].set_title('Combined Focus')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# =============================================================================
# SAMPLING STRATEGIES
# =============================================================================

def sample_by_category(rows, n_per_cat=4, seed=42):
    """Sample images from each label category for balanced visualization.
    
    Categories: biopsy_only, mag_only, both, neither
    
    Returns list of (row, category_name) tuples
    """
    np.random.seed(seed)
    
    # Partition by label combination
    cats = {
        'biopsy':  L(rows).filter(lambda r: r['has_biopsy_tool']==1 and r['has_mag_view']==0),
        'magview': L(rows).filter(lambda r: r['has_biopsy_tool']==0 and r['has_mag_view']==1),
        'both':    L(rows).filter(lambda r: r['has_biopsy_tool']==1 and r['has_mag_view']==1),
        'neither': L(rows).filter(lambda r: r['has_biopsy_tool']==0 and r['has_mag_view']==0),
    }
    
    samples = []
    for name, cat_rows in cats.items():
        if not cat_rows: continue
        n = min(n_per_cat, len(cat_rows))
        idxs = np.random.choice(len(cat_rows), n, replace=False)
        for i in idxs: samples.append((cat_rows[i], name))
    
    return samples

# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

def generate_gradcam_visualizations(model_path, output_dir, db_path, n_samples=16):
    """Generate GradCAM visualizations for sample images.
    
    Creates 4-panel images showing where the model focuses for each class.
    Samples are drawn from each label category for balanced coverage.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    
    # Target layer: last conv layer outputs 7×7×1280 feature maps
    # This is the sweet spot - still has spatial info, but represents high-level features
    # Earlier layers (edges/textures) would be too noisy
    # Later layers (classifier head) have no spatial info left
    target_layer = model.features[-1]
    
    gradcam = GradCAM(model, target_layer)
    
    # Get labeled images from database
    db = get_db(db_path)
    labeled = get_labeled(db)  # Returns train+val, excludes hold-out
    
    if not labeled:
        print("No labeled images found in database")
        return
    
    # Sample from each category
    samples = sample_by_category(labeled, n_per_cat=n_samples//4)
    print(f"Generating GradCAM for {len(samples)} images...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (row, category) in enumerate(progress_bar(samples)):
        thumb_path = Path(row['thumbnail_path'])
        if not thumb_path.exists():
            print(f"  Skipping missing: {thumb_path}")
            continue
        
        # Load and preprocess
        x, img_arr = load_image(thumb_path, device)
        
        # Get predictions
        with torch.no_grad():
            out = model(x)
            probs = torch.sigmoid(out).cpu().numpy()[0]
        
        # Generate heatmaps for each class
        # Note: we generate SEPARATE heatmaps because each class may focus on different regions
        cam_biopsy = gradcam.generate(x, target_class=0)  # Where did it look for biopsy?
        cam_mag = gradcam.generate(x, target_class=1)     # Where did it look for mag view?
        
        # Save visualization
        save_path = output_dir / f'gradcam_{category}_{idx:03d}.png'
        plot_gradcam(img_arr, cam_biopsy, cam_mag, probs, row, save_path)
        
        print(f"  Saved: {save_path.name}")
    
    print(f"\nGradCAM visualizations saved to: {output_dir}")

# =============================================================================
# INTERPRETING GRADCAM OUTPUT
# =============================================================================
#
# GOOD SIGNS:
# - Biopsy heatmap highlights the bright metallic marker region
# - Mag view heatmap highlights the compression plate edges
# - Heatmaps are localized, not diffuse
#
# BAD SIGNS:
# - Heatmap covers the entire image uniformly (model isn't focusing)
# - Heatmap highlights image borders or corners (spurious artifacts)
# - Wrong class heatmap highlights the actual feature (class confusion)
#
# DEBUGGING WORKFLOW:
# 1. Generate GradCAM for CORRECT predictions - verify model looks at right features
# 2. Generate GradCAM for WRONG predictions - understand failure modes
# 3. If model focuses on artifacts (patient ID, scanner borders), you have data bias
# 4. Use insights to guide active learning: prioritize labeling where focus is suspicious

# =============================================================================
# CLI
# =============================================================================

@call_parse
def main(
    model:Path=MODEL_DIR/"model.pth",      # Trained model path
    output_dir:Path=MODEL_DIR/"gradcam",   # Output directory for visualizations
    db:Path=DB_PATH,                       # Labels database
    n_samples:int=16,                      # Total samples (4 per category)
):
    "Generate GradCAM visualizations to understand model attention"
    if not model.exists():
        print(f"Error: Model not found: {model}")
        print("Train a model first: python scripts/03_train.py")
        sys.exit(1)
    
    generate_gradcam_visualizations(model, output_dir, db, n_samples)