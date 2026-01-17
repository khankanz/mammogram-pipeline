#!/usr/bin/env python3
"""Train multi-label classifier from Biopsy Tool and Mag View detection.

Optimized for small datasets:
- EfficientNet-B0 (fewer params than ResNet18)
- Strong data augmentation
- Focal loss for class imbalance
- Gradual unfreezing 
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from fastcore.script import call_parse, Param
from fastcore.basics import store_attr
from fastprogress import progress_bar

# Let;s talk EfficientNet-B0
# This puppy has ~5.3M params; for comparison; ResNet18 has 11.7M, ResNet50 has 25.6M and EfficientNet-B7 has 66M
# Smaller model = faster training, less overfitting on tiny datasets. The dropout before the final layer (dropout=0.4) is aggressive regularization
# 40% of features randomly zeroed during training. Forces the model to not rely on any single feature.
# Dropout: > "The key insight is that at test time, we use all the activations, but we scale them... It's like each mini-batch is training a slightly different model, and at test time we're averaging all of them."

from lib.config import (
    MODEL_DIR, DB_PATH, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED, ensure_dirs
)
from lib.db import get_db, get_train_val, get_labeled
LABEL_COLS = ['has_biopsy_tool', 'has_mag_view']
# =============================================================================
# FOCAL LOSS
# =============================================================================

# Think of this as: stop patting yourself on the back for things you already know. Focus on what confuses you.
    # more precisely: down-weight well-classified examples when pt probability of correct class is high, 1-pt^y term crushes the loss towrads zero
        # focus gradients on hard examples; misclassified or uncertain examples retain full loss magnitude
        # in our case; since ~80% is neither, that's a large imbalance plus the neither cases are easier i.e. no distinctive features to find. Focal loss fits.
        # We might also see this pattern in: medical imaging rare diseases detectin in screening, fraud detection, defect detection in manufacturing 
# Standard BE loss spends most of it's gradient budget on easy 'neither' examples the already nails.
# Story time kids: Focal Loss was introduced by Facebook AI Research in 2017 paper titled "Focal Loss for Dense Object Detection" (the RetinaNet paper)
# The problem they were solving; object detectors look at thousands of candidate regions but MOST are background. Standard cross-entropy treats every example equally
# model spends most of it's learning capacity on easy 'this is obviously background' examples
class FocalLoss(nn.Module):
    "Focal Loss for class imbalance - down-weights easy examples."
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    
    def forward(self, inp, targ):
        bce = F.binary_cross_entropy_with_logits(inp, targ, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1-pt)**self.gamma * bce).mean()
# =============================================================================
# DATASET
# =============================================================================

# ImageNet normalization - normalize to what the pretrained model expects
# imagenet images; when converted to 0-1 float range, happen to have these stats.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(aug=False, sz=IMAGE_SIZE):
    "Get transforms for training (with aug) or validation (without)."
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    # These Augmentations are applied to the training data only
    if aug: return T.Compose([
        T.RandomHorizontalFlip(p=0.5), # mirror left to right and vice versa
        T.RandomRotation(15), # slight rotation; scanner orientation varies
        T.ColorJitter(brightness=0.2, contrast=0.2), # brightness and contrast jitter; different exposures across hospitals
        T.RandomResizedCrop(sz, scale=(0.8, 1.0)), # crop + resize; forces model to find features anywhere
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.2, scale=(0.02, 0.1)), # Black out random rectangles; 'cutout' regularization forces redundancy
    ])
    # For validation; no augmentation; just resize and normalize
    return T.Compose([T.Resize(sz), T.ToTensor(), normalize])

class MultiLabelDS(Dataset):
    "Dataset for multi-label classification with optional augmentation."
    def __init__(self, data, aug=False):
        self.data, self.tfm = data, get_transforms(aug)
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        item = self.data[i]
        img = self.tfm(Image.open(item['path']).convert('RGB'))
        labels = torch.tensor([item['has_biopsy_tool'], item['has_mag_view']], dtype=torch.float32)
        return img, labels
# =============================================================================
# MODEL
# =============================================================================
def create_model(n_out=2, dropout=0.4):
    "Create EfficientNet-B0 for multi-label classification."
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    n_in = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_in, n_out))
    return model

def freeze_backbone(m):
    "Freeze all layers except classifier."
    for p in m.features.parameters(): p.requires_grad = False

def unfreeze_backbone(m):
    "Unfreeze all layers."
    for p in m.parameters(): p.requires_grad = True

# =============================================================================
# TRAINING UTILITIES
# =============================================================================
def build_items(rows):
    "Convert db rows to list of dicts with path and labels."
    items = []
    for r in rows:
        p = Path(r['thumbnail_path'])
        if p.exists(): items.append({'path': p, 'has_biopsy_tool': float(r['has_biopsy_tool']), 'has_mag_view': float(r['has_mag_view'])})
        else: print(f"WARNING: Missing thumbnail {p}") # this should be a log
    return items

def print_class_dist(items):
    "Print class distribution for debugging."
    n = len(items)
    biopsy = sum(1 for d in items if d['has_biopsy_tool'] == 1)
    mag = sum(1 for d in items if d['has_mag_view'] == 1)
    both = sum(1 for d in items if d['has_biopsy_tool'] == 1 and d['has_mag_view'] == 1)
    neither = sum(1 for d in items if d['has_biopsy_tool'] == 0 and d['has_mag_view'] == 0)

    # Should switch to logging
    print(f"Class distribution:")
    print(f"  Biopsy Tool: {biopsy} ({100*biopsy/n:.1f}%)")
    print(f"  Mag View: {mag} ({100*mag/n:.1f}%)")
    print(f"  Both: {both} ({100*both/n:.1f}%)")
    print(f"  Neither: {neither} ({100*neither/n:.1f}%)")

def evaluate(model, dl, device):
    "Evaluate model, return (avg_acc, biopsy_acc, mag_acc)"
    model.eval()
    all_preds, all_targs = [], []
    
    with torch.no_grad():
        for xb, yb in dl:
            out = model(xb.to(device))
            all_preds.append(torch.sigmoid(out).cpu())
            all_targs.append(yb)
    
    preds = torch.cat(all_preds)
    targs = torch.cat(all_targs)
    preds_bin = (preds > 0.5).float()

    biopsy_acc = (preds_bin[:, 0] == targs[:, 0]).float().mean().item()
    mag_acc = (preds_bin[:, 1] == targs[:, 1]).float().mean().item()
    return (biopsy_acc + mag_acc) / 2, biopsy_acc, mag_acc

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

# The why training decisions:
# Backbone outputs a 1280 dims feature vector
# Why weight decay: why it matters
    # **What it does:** Each training step, weights are multiplied by `(1 - lr * weight_decay)` before the gradient update. With `weight_decay=0.01` and `lr=1e-4`, weights shrink by 0.000001 per step.
    # Weight decay pulls all weights toward zero. Features need to "earn" their magnitude by consistently helping across many examples. It's L2 regularization baked into the optimizer.
    # As Jeremy said and we're paraphrasing: > "Weight decay is one of the most important regularization techniques... it's like you're saying to the model 'I don't believe any of your weights should be very big unless you can prove it to me.'"
# **Jeremy's discriminative learning rates:** fastai actually goes further—it uses *different* learning rates for different layers. Earlier layers (edges, textures) get tiny LR. Later layers (high-level features) get larger LR. The head gets the largest. This code uses a simpler uniform LR approach.

def train_model(db_path, output_dir, bs=BATCH_SIZE, epochs=20):
    "Train multi-label classifier with two-phase training."
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    db = get_db(db_path)
    train_rows, val_rows = get_train_val(db)

    if len(train_rows) + len(val_rows) < 10:
        print(f"ERROR: Need at least 10 labeled images, have {len(train_rows) + len(val_rows)}")
        sys.exit(1)
    
    print(f"Training on {len(train_rows) + len(val_rows)} labeled images.")

    train_items, val_items = build_items(train_rows), build_items(val_rows)
    print(f"Found {len(train_items)} train and {len(val_items)} valid thumbnails.")
    print_class_dist(train_items+val_items)

    train_ds, val_ds = MultiLabelDS(train_items, aug=True), MultiLabelDS(val_items, aug=False)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4)

    model = create_model(n_out=2, dropout=0.4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: EfficientNet-B0 (5.3M params)")
    model = model.to(device)

    loss_fn = FocalLoss(alpha=1, gamma=2)
    best_acc, best_state = 0, None
    # Phase 1: Train only classifier head (frozen backbone)
    # EfficientNet-B0 has ImageNet features baked in; edges, textures, shapes. We don't touch those
    # Just train the classifier layer to map those features -> your labels. 
    # High learning rate is fine because we're only updating a small layer
    print(f"\n=== Phase 1: Training classifier head (backbone frozen) ===")
    freeze_backbone(model)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0.01)
    phase1_epochs = min(5, epochs // 3)

    for epoch in range(phase1_epochs):
        model.train()
        train_loss = 0
        for xb, yb in progress_bar(train_dl, leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        val_acc, biopsy_acc, mag_acc = evaluate(model, val_dl, device)
        print(f"Epoch {epoch+1}/{phase1_epochs}: loss ={train_loss/len(train_dl):.4f}, "
            f"biopsy={biopsy_acc:.4f}, mag={mag_acc:.4f}, avg={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"     -> New best!")
    # Phase 2: Fine-tune entire network
    # Now the classifier knows roughly what to look for
    # Unfreeze the backbone and let the whole network adapt to mammogram specific features
    # Lower learning rate by 1e-4 prevents destroying ImageNet features
    # WHY not just train everything from scratch? We'd need 100x more data. Transfer learning is our friend and cheat code for small datasets
    print(f"\n=== Phase 2: Fine-tuning entire network ===")
    unfreeze_backbone(model)

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    # Learning Rate Scheduling; CosineAnnealing 
    # LR starts with 1e-4, smoothly decays; following a cosine curve down to 1e-6. 
    # Translation: big steps to find the right region. Later training is like okay enough exploration, let's settle down into the min.
    # What about linear or step decay? linear is too aggressive at the end, step decay is too abrupt. Consine is goldlocks zone.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - phase1_epochs, eta_min=1e-6)

    for epoch in range(epochs - phase1_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for xb, yb in progress_bar(train_dl, leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # This bad boy here has been my saviour many times in the past. IF a batch produces insane gradients (exploding gradients, not kittens), this caps them at magnitude 1.0. Without this, one bad batch can send your weights to infinity. So that's like weird scanner artifacts, corrupted pixels all that jazz.
            opt.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).float()
            train_correct += (preds == yb).all(dim=1).sum().item()
            train_total += yb.size(0)
        
        sched.step()
        train_acc = train_correct / train_total
        val_acc, biopsy_acc, mag_acc = evaluate(model, val_dl, device)

        print(f"Epoch {epoch+1}/{epochs-phase1_epochs}: "
              f"loss={train_loss/len(train_dl):.4f}, train_acc={train_acc:.4f}, "
              f"biopsy={biopsy_acc:.4f}, mag={mag_acc:.4f}, avg={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best!")
        
    # Load best and save
    model.load_state_dict(best_state)
    model = model.to(device)

    # Final evaluation
    print(f"\n=== Final Validation Results ===")
    model.eval()
    val_preds, val_targs = [], []

    with torch.no_grad():
        for xb, yb in val_dl:
            out = model(xb.to(device))
            val_preds.append(torch.sigmoid(out).cpu())
            val_targs.append(yb)
    
    val_preds = torch.cat(val_preds).numpy()
    val_targs = torch.cat(val_targs).numpy()
    val_preds_bin = (val_preds > 0.5).astype(int)

    print("\nBiopsy Tool:")
    print(classification_report(val_targs[:, 0], val_preds_bin[:, 0], target_names=['No', 'Yes'], zero_division=0))
    print("Mag View:")
    print(classification_report(val_targs[:, 1], val_preds_bin[:, 1], target_names=['No', 'Yes'], zero_division=0))
    
    model_path = output_dir / "model.pth"
    torch.save({'model_state_dict': best_state, 'model_type': 'efficientnet_b0', 'label_cols': LABEL_COLS}, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model_path

# =============================================================================
# PREDICTION UPDATE (ACTIVE LEARNING)
# =============================================================================

# NOW after training, we run predictions on UNLABELED images. These confidence scored feed back into get_unlabeled() in our labeling UI.
# Images with confidence ~= 0.5 -> model is confused -> high-value labels
# Images with confidence ~= 0 or 1 -> model is confident -> check for systematic errors
# THIS is the active learning loop closing. The next labeling session shows us what the model needs, not random images.

def update_predictions(db_path, model_path, bs=BATCH_SIZE):
    "Run predictions on unlabeled images to feed active learning loop."
    db = get_db(db_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(n_out=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    unlabeled = list(db["labels"].rows_where(
        "(has_biopsy_tool IS NULL OR has_mag_view IS NULL) "
        "AND (split IS NULL OR split IN ('train', 'val'))"
    ))
    
    if not unlabeled:
        print("No unlabeled images to predict")
        return
    
    print(f"\nPredicting on {len(unlabeled)} unlabeled images...")
    
    tfm = get_transforms(aug=False)
    now = datetime.now().isoformat()
    updated = 0
    
    with torch.no_grad():
        for row in progress_bar(unlabeled):
            thumb_path = Path(row["thumbnail_path"])
            if not thumb_path.exists():
                print(f"Warning: Missing thumbnail {thumb_path}")
                continue
            
            try:
                img = tfm(Image.open(thumb_path).convert('RGB')).unsqueeze(0).to(device)
                probs = torch.sigmoid(model(img)).cpu().numpy()[0]
                
                if row.get("split") == "test":
                    print(f"Warning: Skipping hold-out image id {row['id']}")
                else:
                    db["labels"].update(row["id"], {
                        "confidence_biopsy": float(probs[0]),
                        "confidence_mag": float(probs[1]),
                        "predicted_at": now,
                    })
                    updated += 1
            except Exception as e:
                print(f"Warning: Failed to predict {thumb_path}: {e}")
    
    print(f"Updated predictions for {updated} images")

# =============================================================================
# HOLDOUT EVALUATION
# =============================================================================

# The final exam
# Remember that 10% holdout from insert_image()? It's been sitting untouched through all our iterations
# This is our unbiased performance estimate. Run it once when you're done iterating - not during development

def evaluate_holdout(db_path, model_path, bs=BATCH_SIZE):
    "Final evaluation on pristine hold-out set. Run ONCE at the end."
    db = get_db(db_path)
    holdout_rows = get_labeled(db, split="test")
    if not holdout_rows: raise ValueError("No labeled hold-out images!")
    
    items = build_items(holdout_rows)
    if not items: raise ValueError("No hold-out thumbnails found on disk.")
    
    holdout_ds = MultiLabelDS(items, aug=False)
    holdout_dl = DataLoader(holdout_ds, batch_size=bs, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(n_out=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    all_preds, all_targs = [], []
    with torch.no_grad():
        for xb, yb in holdout_dl:
            out = model(xb.to(device))
            all_preds.append(torch.sigmoid(out).cpu())
            all_targs.append(yb)
    
    preds = torch.cat(all_preds).numpy()
    targs = torch.cat(all_targs).numpy()
    preds_bin = (preds > 0.5).astype(int)
    
    print("\n=== Hold-out Evaluation (Test Split) ===")
    print(f"Images evaluated: {len(items)}")
    print("\nBiopsy Tool:")
    print(classification_report(targs[:, 0], preds_bin[:, 0], target_names=["No", "Yes"], zero_division=0))
    print("Mag View:")
    print(classification_report(targs[:, 1], preds_bin[:, 1], target_names=["No", "Yes"], zero_division=0))
    
    biopsy_acc = (preds_bin[:, 0] == targs[:, 0]).mean().item()
    mag_acc    = (preds_bin[:, 1] == targs[:, 1]).mean().item()
    avg_acc    = (biopsy_acc + mag_acc) / 2
    print(f"Hold-out accuracy: biopsy={biopsy_acc:.4f}, mag={mag_acc:.4f}, avg={avg_acc:.4f}")
    
    return {"biopsy_acc": biopsy_acc, "mag_acc": mag_acc, "avg_acc": avg_acc}

# =============================================================================
# CLI
# =============================================================================

@call_parse
def main(
    db:Path=DB_PATH,                           # SQLite database path
    output_dir:Path=MODEL_DIR,                 # Output directory for model
    model:Path=MODEL_DIR/"model.pth",          # Trained model path for holdout eval
    batch_size:int=BATCH_SIZE,                 # Batch size
    epochs:int=20,                             # Training epochs
    no_predict:bool=False,                     # Skip prediction update
    evaluate_holdout_flag:bool=False,          # Evaluate on holdout and exit
):
    "Train multi-label DICOM classifier with active learning support."
    if evaluate_holdout_flag:
        evaluate_holdout(db, model, batch_size)
        return
    
    model_path = train_model(db_path=db, output_dir=output_dir, bs=batch_size, epochs=epochs)
    
    if not no_predict:
        update_predictions(db, model_path, batch_size)
    
    print("\nTraining complete!")
    print("Next steps:")
    print("  1. Run GradCAM: python scripts/06_gradcam.py")
    print("  2. Export ONNX: python scripts/04_export.py")

