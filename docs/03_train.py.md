# Training Pipeline: Multi-Label Classification with Active Learning

This script trains a multi-label classifier for biopsy tool and magnification view detection. It's optimized for small datasets—the kind you get when you're actively labeling and iterating, not when you have 100K images.

## Why EfficientNet-B0?

```python
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
```

Parameter counts matter when you have <500 labeled images:

| Model | Parameters |
|-------|------------|
| EfficientNet-B0 | 5.3M |
| ResNet18 | 11.7M |
| ResNet50 | 25.6M |
| EfficientNet-B7 | 66M |

Smaller model = faster training, less overfitting on tiny datasets. The aggressive dropout (0.4) before the final layer zeros out 40% of features during training, forcing the model to not rely on any single feature.

As Jeremy Howard put it: "The key insight is that at test time, we use all the activations, but we scale them... It's like each mini-batch is training a slightly different model, and at test time we're averaging all of them."

---

## Focal Loss: Stop Rewarding Easy Wins

Standard cross-entropy spends most of its gradient budget on easy examples—the 80% of images that are "neither biopsy nor mag view" and have no distinctive features to find.

```python
class FocalLoss(nn.Module):
    "Focal Loss for class imbalance - down-weights easy examples."
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    
    def forward(self, inp, targ):
        bce = F.binary_cross_entropy_with_logits(inp, targ, reduction='none')
        pt = torch.exp(-bce)  # probability of correct class
        return (self.alpha * (1-pt)**self.gamma * bce).mean()
```

Think of it as: stop patting yourself on the back for things you already know. Focus on what confuses you.

More precisely:
- When `pt` (probability of correct class) is high, the `(1-pt)^γ` term crushes the loss toward zero
- Misclassified or uncertain examples retain full loss magnitude
- Gradients focus on hard examples

Focal Loss was introduced by Facebook AI Research in 2017 ("Focal Loss for Dense Object Detection"—the RetinaNet paper). The problem they solved: object detectors look at thousands of candidate regions, but most are background. Standard cross-entropy treats every example equally, so the model spends most of its learning capacity on easy "this is obviously background" cases.

Same pattern applies to: medical imaging rare disease detection, fraud detection, manufacturing defect detection—anywhere you have massive class imbalance.

---

## Data Augmentation: Fake Variety

With small datasets, augmentation is crucial. We simulate the variation you'd see across different scanners and hospitals:

```python
def get_transforms(aug=False, sz=IMAGE_SIZE):
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if aug: return T.Compose([
        T.RandomHorizontalFlip(p=0.5),           # Mirror—scanner orientation varies
        T.RandomRotation(15),                     # Slight rotation
        T.ColorJitter(brightness=0.2, contrast=0.2),  # Different exposures
        T.RandomResizedCrop(sz, scale=(0.8, 1.0)),    # Forces model to find features anywhere
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.2, scale=(0.02, 0.1)),    # "Cutout" regularization
    ])
    
    # Validation: no augmentation
    return T.Compose([T.Resize(sz), T.ToTensor(), normalize])
```

`RandomErasing` (cutout) blacks out random rectangles—forces redundancy in the learned features. If part of the biopsy tool is occluded, the model should still recognize it.

The ImageNet normalization (`IMAGENET_MEAN`, `IMAGENET_STD`) matches what the pretrained model expects. ImageNet images, when converted to 0-1 float range, happen to have these specific statistics.

---

## Dataset: Simple and Direct

```python
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
```

Multi-label means both outputs are independent binary classifications. An image can be biopsy=1, mag=0 or biopsy=1, mag=1 or any combination. We use `float32` labels because BCE loss expects floats.

---

## Model Architecture

```python
def create_model(n_out=2, dropout=0.4):
    "Create EfficientNet-B0 for multi-label classification."
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    n_in = model.classifier[1].in_features  # 1280 for EfficientNet-B0
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_in, n_out))
    return model
```

The backbone outputs a 1280-dimensional feature vector. We replace the original 1000-class ImageNet head with: dropout → 2-class output.

Freezing utilities for two-phase training:

```python
def freeze_backbone(m):
    "Freeze all layers except classifier."
    for p in m.features.parameters(): p.requires_grad = False

def unfreeze_backbone(m):
    "Unfreeze all layers."
    for p in m.parameters(): p.requires_grad = True
```

---

## Two-Phase Training

### Phase 1: Train Classifier Head Only

```python
print(f"\n=== Phase 1: Training classifier head (backbone frozen) ===")
freeze_backbone(model)

opt = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-3, 
    weight_decay=0.01
)
phase1_epochs = min(5, epochs // 3)
```

EfficientNet-B0 has ImageNet features baked in—edges, textures, shapes. We don't touch those. Just train the classifier layer to map those features → your labels.

High learning rate (1e-3) is fine because we're only updating a small layer.

### Phase 2: Fine-tune Entire Network

```python
print(f"\n=== Phase 2: Fine-tuning entire network ===")
unfreeze_backbone(model)

if best_state:
    model.load_state_dict(best_state)
    model = model.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - phase1_epochs, eta_min=1e-6)
```

Now the classifier knows roughly what to look for. Unfreeze the backbone and let the whole network adapt to mammogram-specific features.

Lower learning rate (1e-4) prevents destroying ImageNet features.

**Why not train everything from scratch?** You'd need 100x more data. Transfer learning is your cheat code for small datasets.

---

## Weight Decay: Earn Your Magnitude

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

What it does: each training step, weights are multiplied by `(1 - lr * weight_decay)` before the gradient update. With `weight_decay=0.01` and `lr=1e-4`, weights shrink by 0.000001 per step.

Weight decay pulls all weights toward zero. Features need to "earn" their magnitude by consistently helping across many examples. It's L2 regularization baked into the optimizer.

As Jeremy Howard said (paraphrasing): "Weight decay is one of the most important regularization techniques... it's like you're saying to the model 'I don't believe any of your weights should be very big unless you can prove it to me.'"

**Note on discriminative learning rates:** fastai actually goes further—different learning rates for different layers. Earlier layers (edges, textures) get tiny LR. Later layers (high-level features) get larger LR. The head gets the largest. This code uses a simpler uniform LR approach.

---

## Learning Rate Scheduling: Cosine Annealing

```python
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - phase1_epochs, eta_min=1e-6)
```

LR starts at 1e-4, smoothly decays following a cosine curve down to 1e-6.

Translation: early training takes big steps to find the right region. Later training says "okay, enough exploration, let's settle into the minimum."

Why cosine over linear or step decay? Linear is too aggressive at the end. Step decay is too abrupt. Cosine is the Goldilocks zone.

---

## Gradient Clipping: Insurance Against Bad Batches

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

This has saved me many times. If a batch produces insane gradients (exploding gradients, not kittens), this caps them at magnitude 1.0.

Without this, one bad batch can send your weights to infinity. Causes: weird scanner artifacts, corrupted pixels, that one image that's somehow all NaN.

---

## Evaluation

```python
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
```

We threshold at 0.5 for binary predictions. The average of both accuracies gives a single number to track.

Final validation uses sklearn's `classification_report` for precision/recall/F1 per class.

---

## Active Learning: Closing the Loop

After training, we run predictions on UNLABELED images:

```python
def update_predictions(db_path, model_path, bs=BATCH_SIZE):
    "Run predictions on unlabeled images to feed active learning loop."
    # ...
    unlabeled = list(db["labels"].rows_where(
        "(has_biopsy_tool IS NULL OR has_mag_view IS NULL) "
        "AND (split IS NULL OR split IN ('train', 'val'))"
    ))
    # ...
    db["labels"].update(row["id"], {
        "confidence_biopsy": float(probs[0]),
        "confidence_mag": float(probs[1]),
        "predicted_at": now,
    })
```

These confidence scores feed back into `get_unlabeled()` in the labeling UI:

- **Confidence ≈ 0.5** → Model is confused → High-value labels
- **Confidence ≈ 0 or 1** → Model is confident → Check for systematic errors

This is the active learning loop closing. The next labeling session shows you what the model *needs*, not random images.

---

## Hold-out Evaluation: The Final Exam

```python
def evaluate_holdout(db_path, model_path, bs=BATCH_SIZE):
    "Final evaluation on pristine hold-out set. Run ONCE at the end."
    db = get_db(db_path)
    holdout_rows = get_labeled(db, split="test")
    # ...
```

Remember that 10% hold-out from `insert_image()`? It's been sitting untouched through all your iterations.

This is your unbiased performance estimate. Run it **once** when you're done iterating—not during development. If you peek at hold-out during development and adjust accordingly, it's no longer unbiased.

---

## CLI

```python
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
```

Usage:

```bash
# Train and update predictions
python scripts/03_train.py

# Train only, skip prediction update
python scripts/03_train.py --no_predict

# Evaluate on hold-out (run once at the end)
python scripts/03_train.py --evaluate_holdout_flag
```

---

## The Full Active Learning Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. LABEL (02_label_server.py)                                   │
│    - Active learning selects high-value images                  │
│    - You press 1/2/3/4 keys rapidly                             │
│    - Labels saved to SQLite                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. TRAIN (03_train.py)                                          │
│    - Two-phase: frozen backbone → fine-tune                     │
│    - Focal loss handles class imbalance                         │
│    - Best model saved to disk                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. PREDICT (automatic after training)                           │
│    - Run inference on unlabeled images                          │
│    - Store confidence scores in SQLite                          │
│    - These scores guide next labeling session                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                        Back to step 1
```

Each iteration: label ~50 images, train, predict, repeat. The model gets smarter about what it needs to learn, and your labels become more valuable.

---

That's the training pipeline. EfficientNet-B0 for small datasets, focal loss for imbalance, two-phase training for transfer learning, and predictions that close the active learning loop.