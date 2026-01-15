# SQLite Database Layer: Active Learning Infrastructure

The database layer isn't just storage—it's the backbone of an active learning loop. It tracks what's been labeled, what needs labeling, and maintains a pristine hold-out set that never leaks into training.

## The Schema

Single source of truth. One table, one place to look:

```python
_schema = dict(
    id=int, filename=str, study_id=str, thumbnail_path=str, frame_number=int,
    has_biopsy_tool=int, has_mag_view=int, labeled_at=str, 
    confidence_biopsy=float, confidence_mag=float, predicted_at=str, split=str
)
```

Two label columns (`has_biopsy_tool`, `has_mag_view`), two confidence columns (model predictions), and a `split` column that determines train/val/test assignment. Everything else is metadata.

Indexes are strategic—we query by filename (uniqueness), study_id (grouping), confidence scores (active learning), and split (training):

```python
_indexes = [
    (["filename"], {"unique": True}),
    (["study_id"], {}),
    (["has_biopsy_tool", "has_mag_view"], {}),
    (["confidence_biopsy"], {}),
    (["confidence_mag"], {}),
    (["split"], {}),
]
```

---

## The Split Strategy: Two Hashes, Two Decisions

Here's where it gets interesting. We need deterministic, reproducible splits—but we also need to make *two independent decisions* per image:

1. **Is this image hold-out?** (test set, locked forever)
2. **If not hold-out, is it train or val?** (assigned when first labeled)

One hash won't cut it. If we used the same hash for both decisions, the bottom 10% by hash would be hold-out AND would have a bias toward one split. We need independence.

```python
def _hash_frac(s, seed=RANDOM_SEED):
    "Deterministic fraction 0-1 from string. Same input = same output."
    return int(hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

def _split_for(filename):
    "Train/val assignment via hash. Prefix ensures different roll than hold-out selection."
    return "val" if _hash_frac(f"trainval:{filename}") < VALID_PCT else "train"
```

The `trainval:` prefix is the key. Same filename, different hash input, independent decision. `image_001.dcm` might hash to 0.08 for hold-out selection (bottom 10% → test) but hash to 0.73 for train/val (→ train). Different rolls, same determinism.

---

## Hold-out Assignment: Lock It Early

The hold-out set is sacred. Once assigned, it never changes. We assign it *before* any labeling happens:

```python
def _assign_holdout(db):
    "Mark bottom HOLDOUT_PCT of fresh images as test. Deterministic via hash."
    rows = list(db["labels"].rows_where(
        "split IS NULL AND has_biopsy_tool IS NULL AND has_mag_view IS NULL"))
    if not rows: return 0
    
    n = int(round(len(rows) * HOLDOUT_PCT))
    if n <= 0: return 0
    
    ranked = sorted(rows, key=lambda r: (_hash_frac(r["filename"]), r["filename"]))
    for r in ranked[:n]: db["labels"].update(r["id"], {"split": "test"})
    return n
```

This looks at all unlabeled, unassigned images, ranks them by hash, and marks the bottom 10% as `test`. Everyone else stays `split=NULL`—available for labeling but not yet assigned to train/val.

---

## The Lifecycle: From Ingestion to Training

Here's how an image flows through the system:

```
DAY 1: Preprocess 1000 DICOMs
       ┌─────────────────────────────────────────┐
       │ All 1000 images: split=NULL, labels=NULL│
       └─────────────────────────────────────────┘
                          │
                          ▼ _assign_holdout()
       ┌─────────────────────────────────────────┐
       │ 100 images: split="test"                │ ← LOCKED FOREVER
       │ 900 images: split=NULL                  │ ← Available for labeling
       └─────────────────────────────────────────┘

DAY 2: Label 50 images via UI
       ┌─────────────────────────────────────────┐
       │ 100 images: split="test" (untouched)    │
       │ 50 images: split="train" or "val"       │ ← Just assigned by hash
       │ 850 images: split=NULL (still waiting)  │
       └─────────────────────────────────────────┘

DAY 7: Final state after 500 labels
       ┌─────────────────────────────────────────┐
       │ 100 images: split="test" (STILL untouched) │
       │ 500 images: split="train" (~400) or "val" (~100) │
       │ 400 images: split=NULL (never labeled)  │
       └─────────────────────────────────────────┘
```

The train/val assignment happens lazily—only when you actually label an image:

```python
def _ensure_split(db, image_id):
    "Assign train/val on first label. Hold-out and existing splits untouched."
    row = db["labels"].get(image_id)
    if row.get("split") in ("test", "train", "val"): return
    db["labels"].update(image_id, {"split": _split_for(row["filename"])})
```

Already hold-out? Don't touch. Already assigned? Don't change. Otherwise, flip the (deterministic) coin.

---

## Safety Rails: No Leaks Allowed

Hold-out contamination ruins everything. One function, one job—crash loudly if something's wrong:

```python
def _no_holdout(rows):
    "Safety check: crash if hold-out leaked into training data"
    assert all(r.get("split") != "test" for r in rows), "Hold-out leak!"
```

This gets called in `get_labeled()` and `get_train_val()`. If hold-out data somehow ended up in your training set, you'll know immediately.

---

## WAL Mode: Concurrent Reads and Writes

SQLite's default journaling blocks readers during writes. WAL (Write-Ahead Logging) fixes this:

```python
def get_db(path=None):
    db = Database(str(ifnone(path, DB_PATH)))
    db.execute("PRAGMA journal_mode=WAL")
    # ...
```

How it works:

```
┌─────────────────────────────────────────────────────────────┐
│                        READ FLOW                            │
│                                                             │
│  Query: SELECT row 5                                        │
│           ↓                                                 │
│  Check WAL index (-shm): "row 5 modified?"                  │
│           ↓                                                 │
│     ┌─────┴─────┐                                           │
│    YES          NO                                          │
│     ↓            ↓                                          │
│  Read latest   Read from                                    │
│  from WAL      main .db                                     │
└─────────────────────────────────────────────────────────────┘
```

Writes go to the WAL file immediately. Reads check WAL first, fall back to main DB. Checkpoints merge WAL into the main file periodically. Net effect: your labeling UI can read while predictions are being written.

---

## Active Learning: Smart Sample Selection

This is where the ML loop closes. `get_unlabeled()` doesn't just grab random images—it balances three strategies:

```python
def get_unlabeled(db, limit=1, exploration_rate=EXPLORATION_RATE, rng=None):
    r = rng.random()
    if r < exploration_rate:
        # 10% exploration: random
        # ...
    elif r < exploration_rate + 0.45:
        # 45% uncertain: closest to 0.5
        return list(db["labels"].rows_where(
            where, order_by="ABS(COALESCE(confidence_biopsy, 0.5) - 0.5) ASC", limit=limit))
    else:
        # 45% confident: closest to 0.0 or 1.0
        return list(db["labels"].rows_where(
            where, order_by="ABS(COALESCE(confidence_biopsy, 0.5) - 0.5) DESC", limit=limit))
```

- **10% random**: Exploration. Prevents the model from getting stuck in local optima.
- **45% uncertain**: Confidence near 0.5. The model is confused—your label has maximum information value.
- **45% confident**: Confidence near 0 or 1. The model is sure—but *wrong* confident predictions are gold. This catches systematic errors.

The iteration loop looks like:

```
┌─────────────────────────────────────────────────────────────────┐
│ ITERATION 1                                                      │
├─────────────────────────────────────────────────────────────────┤
│  1000 images total                                               │
│  ├── 100 hold-out (split="test", untouched forever)              │
│  └── 900 available (split=NULL, unlabeled)                       │
│                                                                  │
│  Label 50 random images → ~40 train, ~10 val                     │
│  Train model, validate → 67% accuracy                            │
│  Run predictions on 850 UNLABELED images                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ITERATION 2                                                      │
├─────────────────────────────────────────────────────────────────┤
│  get_unlabeled() samples from 850 unlabeled:                     │
│  ├── 10% random                                                  │
│  ├── 45% uncertain (confidence ≈ 0.5)                            │
│  └── 45% confident (confidence ≈ 0.0 or 1.0) ← catches errors    │
│                                                                  │
│  Label 50 more → now 100 labeled (80 train, 20 val)              │
│  Train model, validate → 78% accuracy                            │
│  Run predictions on 800 UNLABELED images                         │
└─────────────────────────────────────────────────────────────────┘
```

Key insight: predictions run on *unlabeled* images only. Train and val sets get labels, not predictions. The confidence scores guide which unlabeled images to label next.

---

## Public API

**Database setup:**
```python
db = get_db()  # Init schema, assign hold-outs, enable WAL
```

**Image ingestion:**
```python
insert_image(db, filename, study_id, thumbnail_path, frame_number=0)
# Skips duplicates, immediately evaluates for hold-out
```

**Labeling workflow:**
```python
rows = get_unlabeled(db, limit=10)  # Active learning selection
set_labels(db, image_id, biopsy=1, mag=0)  # Assigns split on first label
```

**Training:**
```python
train, val = get_train_val(db)  # Safety-checked, no hold-out leaks
```

**Prediction updates:**
```python
update_predictions(db, [{"id": 5, "confidence_biopsy": 0.92, "confidence_mag": 0.15}])
# Skips hold-out and already-labeled images
```

**Stats:**
```python
get_stats(db)  # Counts for dashboard
```

---

That's the database layer. Deterministic splits, active learning sampling, and safety rails that crash loudly if you mess up. The hold-out stays pure, the training loop stays informed.
