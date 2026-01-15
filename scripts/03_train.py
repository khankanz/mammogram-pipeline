#!/usr/bin/env python3
"""Train multi-label classifier for Biopsy Tool and Mag View detection.

Optimized for small datasets:
- EfficientNet-B0 (fewer parameters than ResNet18)
- Strong data augmentation
- Focal loss for class imbalance
- Gradual unfreezing
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from lib.config import (
    MODEL_DIR, DB_PATH,
    BATCH_SIZE, VALID_PCT, RANDOM_SEED,
    ensure_dirs
)
from lib.db import get_db, get_train_val_split, get_labeled_images

LABEL_COLS = ['has_biopsy_tool', 'has_mag_view']
IMAGE_SIZE = 224  # EfficientNet-B0 default


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance - down-weights easy examples."""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class MultiLabelDataset(Dataset):
    """Dataset with augmentation for multi-label classification."""

    def __init__(self, data, augment=False):
        self.data = data
        self.augment = augment

        # ImageNet normalization
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if augment:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
                T.ToTensor(),
                self.normalize,
                T.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Cutout
            ])
        else:
            self.transform = T.Compose([
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                self.normalize,
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['path']).convert('RGB')
        img = self.transform(img)
        labels = torch.tensor([item['has_biopsy_tool'], item['has_mag_view']], dtype=torch.float32)
        return img, labels


def create_model(num_classes=2, dropout=0.4):
    """Create EfficientNet-B0 for multi-label classification."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, num_classes)
    )
    return model


def freeze_backbone(model):
    """Freeze all layers except classifier."""
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model):
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True


def train_model(db_path: Path, output_dir: Path, batch_size: int = BATCH_SIZE,
                epochs: int = 20) -> Path:
    """Train multi-label classifier with optimizations for small datasets."""
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    db = get_db(db_path)
    train_rows, val_rows = get_train_val_split(db)
    labeled = train_rows + val_rows

    if len(labeled) < 10:
        print(f"Error: Need at least 10 labeled images, have {len(labeled)}")
        sys.exit(1)

    print(f"Training on {len(labeled)} labeled images")

    # Build dataset
    def build_dataset(rows):
        items = []
        for row in rows:
            thumb_path = Path(row["thumbnail_path"])
            if thumb_path.exists():
                items.append({
                    'path': thumb_path,
                    'has_biopsy_tool': float(row['has_biopsy_tool']),
                    'has_mag_view': float(row['has_mag_view']),
                })
            else:
                print(f"Warning: Missing thumbnail {thumb_path}")
        return items

    train_data = build_dataset(train_rows)
    valid_data = build_dataset(val_rows)

    print(f"Found {len(train_data)} train and {len(valid_data)} valid thumbnails")

    # Class distribution
    biopsy_yes = sum(1 for d in data if d['has_biopsy_tool'] == 1)
    mag_yes = sum(1 for d in data if d['has_mag_view'] == 1)
    both_yes = sum(1 for d in data if d['has_biopsy_tool'] == 1 and d['has_mag_view'] == 1)
    neither = sum(1 for d in data if d['has_biopsy_tool'] == 0 and d['has_mag_view'] == 0)

    print(f"Class distribution:")
    print(f"  Biopsy Tool: {biopsy_yes} ({100*biopsy_yes/len(data):.1f}%)")
    print(f"  Mag View: {mag_yes} ({100*mag_yes/len(data):.1f}%)")
    print(f"  Both: {both_yes} ({100*both_yes/len(data):.1f}%)")
    print(f"  Neither: {neither} ({100*neither/len(data):.1f}%)")

    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}")

    # Create dataloaders with augmentation for training
    train_ds = MultiLabelDataset(train_data, augment=True)
    valid_ds = MultiLabelDataset(valid_data, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model
    model = create_model(num_classes=2, dropout=0.4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: EfficientNet-B0 (5.3M params)")
    model = model.to(device)

    # Focal loss for class imbalance
    loss_fn = FocalLoss(alpha=1, gamma=2)

    # Phase 1: Train only classifier head (frozen backbone)
    print(f"\n=== Phase 1: Training classifier head (backbone frozen) ===")
    freeze_backbone(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=0.01
    )

    best_acc = 0
    best_model_state = None
    phase1_epochs = min(5, epochs // 3)

    for epoch in range(phase1_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        val_acc, biopsy_acc, mag_acc = evaluate(model, valid_dl, device)
        print(f"Epoch {epoch+1}/{phase1_epochs}: loss={train_loss/len(train_dl):.4f}, "
              f"biopsy={biopsy_acc:.4f}, mag={mag_acc:.4f}, avg={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best!")

    # Phase 2: Fine-tune entire network
    print(f"\n=== Phase 2: Fine-tuning entire network ===")
    unfreeze_backbone(model)

    # Load best from phase 1
    if best_model_state:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - phase1_epochs, eta_min=1e-6
    )

    for epoch in range(epochs - phase1_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).all(dim=1).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_acc = train_correct / train_total

        # Validation
        val_acc, biopsy_acc, mag_acc = evaluate(model, valid_dl, device)

        print(f"Epoch {epoch+1}/{epochs-phase1_epochs}: "
              f"loss={train_loss/len(train_dl):.4f}, train_acc={train_acc:.4f}, "
              f"biopsy={biopsy_acc:.4f}, mag={mag_acc:.4f}, avg={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best!")

    # Load best model
    model.load_state_dict(best_model_state)
    model = model.to(device)

    # Final evaluation
    print("\n=== Final Validation Results ===")
    model.eval()
    val_preds, val_targets = [], []

    with torch.no_grad():
        for images, labels in valid_dl:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            val_preds.append(preds.cpu())
            val_targets.append(labels)

    val_preds = torch.cat(val_preds).numpy()
    val_targets = torch.cat(val_targets).numpy()
    val_preds_binary = (val_preds > 0.5).astype(int)

    print("\nBiopsy Tool:")
    print(classification_report(val_targets[:, 0], val_preds_binary[:, 0],
                                target_names=['No', 'Yes'], zero_division=0))

    print("Mag View:")
    print(classification_report(val_targets[:, 1], val_preds_binary[:, 1],
                                target_names=['No', 'Yes'], zero_division=0))

    # Save model
    model_path = output_dir / "model.pth"
    torch.save({
        'model_state_dict': best_model_state,
        'model_type': 'efficientnet_b0',
        'label_cols': LABEL_COLS,
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    return model_path


def evaluate(model, dataloader, device):
    """Evaluate model and return accuracies."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())
            all_targets.append(labels)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    preds_binary = (preds > 0.5).float()

    biopsy_acc = (preds_binary[:, 0] == targets[:, 0]).float().mean().item()
    mag_acc = (preds_binary[:, 1] == targets[:, 1]).float().mean().item()
    avg_acc = (biopsy_acc + mag_acc) / 2

    return avg_acc, biopsy_acc, mag_acc


def update_predictions(db_path: Path, model_path: Path, batch_size: int = BATCH_SIZE):
    """Run predictions on unlabeled images."""
    db = get_db(db_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Get unlabeled images
    unlabeled = list(db["labels"].rows_where(
        "(has_biopsy_tool IS NULL OR has_mag_view IS NULL) "
        "AND (split IS NULL OR split IN ('train', 'val'))"
    ))

    if not unlabeled:
        print("No unlabeled images to predict")
        return

    print(f"\nPredicting on {len(unlabeled)} unlabeled images...")

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), normalize])

    now = datetime.now().isoformat()
    updated = 0

    with torch.no_grad():
        for row in unlabeled:
            thumb_path = Path(row["thumbnail_path"])
            if not thumb_path.exists():
                print(f"Warning: Missing thumbnail {thumb_path}")
                continue

            try:
                img = Image.open(thumb_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)

                outputs = model(img_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()[0]

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


def evaluate_holdout(db_path: Path, model_path: Path, batch_size: int = BATCH_SIZE) -> dict:
    """Final evaluation on pristine hold-out set. Run ONCE at the end."""
    db = get_db(db_path)
    holdout_rows = get_labeled_images(db, split="test")
    if not holdout_rows:
        raise ValueError("No labeled hold-out images!")

    # Build dataset (no augmentation)
    items = []
    for row in holdout_rows:
        thumb_path = Path(row["thumbnail_path"])
        if thumb_path.exists():
            items.append({
                "path": thumb_path,
                "has_biopsy_tool": float(row["has_biopsy_tool"]),
                "has_mag_view": float(row["has_mag_view"]),
            })
        else:
            print(f"Warning: Missing thumbnail {thumb_path}")

    if not items:
        raise ValueError("No hold-out thumbnails found on disk.")

    holdout_ds = MultiLabelDataset(items, augment=False)
    holdout_dl = DataLoader(holdout_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in holdout_dl:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())
            all_targets.append(labels)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    preds_binary = (preds > 0.5).astype(int)

    print("\n=== Hold-out Evaluation (Test Split) ===")
    print(f"Images evaluated: {len(items)}")
    print("\nBiopsy Tool:")
    print(classification_report(targets[:, 0], preds_binary[:, 0],
                                target_names=["No", "Yes"], zero_division=0))
    print("Mag View:")
    print(classification_report(targets[:, 1], preds_binary[:, 1],
                                target_names=["No", "Yes"], zero_division=0))

    biopsy_acc = (preds_binary[:, 0] == targets[:, 0]).mean().item()
    mag_acc = (preds_binary[:, 1] == targets[:, 1]).mean().item()
    avg_acc = (biopsy_acc + mag_acc) / 2
    print(f"Hold-out accuracy: biopsy={biopsy_acc:.4f}, mag={mag_acc:.4f}, avg={avg_acc:.4f}")

    return {
        "biopsy_acc": biopsy_acc,
        "mag_acc": mag_acc,
        "avg_acc": avg_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Train multi-label DICOM classifier")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "model.pth",
                        help="Trained model path for hold-out evaluation")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--no-predict", action="store_true")
    parser.add_argument("--evaluate-holdout", action="store_true",
                        help="Evaluate on labeled hold-out set and exit")

    args = parser.parse_args()

    if args.evaluate_holdout:
        evaluate_holdout(args.db, args.model, args.batch_size)
        return

    model_path = train_model(
        db_path=args.db,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    if not args.no_predict:
        update_predictions(args.db, model_path, args.batch_size)

    print("\nTraining complete!")
    print("Next steps:")
    print("  1. Run GradCAM: python scripts/06_gradcam.py")
    print("  2. Export ONNX: python scripts/04_export.py")


if __name__ == "__main__":
    main()
