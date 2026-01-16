"""SQLite database helpers using sqlite-utils."""

from sqlite_utils import Database
from datetime import datetime
import hashlib, logging, random
from .config import DB_PATH, HOLDOUT_PCT, EXPLORATION_RATE, RANDOM_SEED, VALID_PCT
from fastcore.basics import ifnone

logger = logging.getLogger(__name__)

# Schema: single source of truth
_schema = dict(id=int, filename=str, study_id=str, thumbnail_path=str, frame_number=int,
    has_biopsy_tool=int, has_mag_view=int, labeled_at=str, confidence_biopsy=float,
    confidence_mag=float, predicted_at=str, split=str)

_indexes = [
    (["filename"], {"unique": True}),
    (["study_id"], {}),
    (["has_biopsy_tool", "has_mag_view"], {}),
    (["confidence_biopsy"], {}),
    (["confidence_mag"], {}),
    (["split"], {}),
]

def _init_db(db):
    "Create table and indexes if needed"
    if "labels" in db.table_names(): return
    db["labels"].create(_schema, pk="id")
    for cols, kw in _indexes: db["labels"].create_index(cols, **kw)

def _hash_frac(s, seed=RANDOM_SEED):
    "Deterministic fraction 0-1 from string. Same input = same output."
    return int(hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

def _split_for(filename):
    "Train/val assignment via hash. Prefix ensures different roll than hold-out selection."
    return "val" if _hash_frac(f"trainval:{filename}") < VALID_PCT else "train"

# Assigns train/val split on first label. Two hash decisions happen at different times:
#   1st hash (at insert_image): hold-out selection - per-image check if hash < HOLDOUT_PCT → "test"
#   2nd hash (here): train/val split - uses "trainval:" prefix for independent coin flip
# Both hashes are deterministic per filename, but prefixes ensure different outcomes.
def _ensure_split(db, image_id):
    "Assign train/val on first label. Hold-out and existing splits untouched."
    row = db["labels"].get(image_id)
    if row.get("split") in ("test", "train", "val"): return
    db["labels"].update(image_id, {"split": _split_for(row["filename"])})
# Before I give you any training data, I want to make sure I didn't give you any hold-out images
# IF this fires, something's wrong with logic


# DAY 1: Preprocess 1000 DICOMs
#        ┌─────────────────────────────────────────┐
#        │ All 1000 images: split=NULL, labels=NULL│
#        └─────────────────────────────────────────┘
#                           │
#                           ▼ assign_holdout_splits()
#        ┌─────────────────────────────────────────┐
#        │ 100 images: split="test"                │ ← LOCKED FOREVER
#        │ 900 images: split=NULL                  │ ← Available for labeling
#        └─────────────────────────────────────────┘

# DAY 2: Label 50 images via UI
#        ┌─────────────────────────────────────────┐
#        │ 100 images: split="test" (untouched)    │
#        │ 50 images: split="train" or "val"       │ ← Just assigned by hash
#        │ 850 images: split=NULL (still waiting)  │
#        └─────────────────────────────────────────┘

# DAY 3: Train model, get predictions, label 50 more
#        ┌─────────────────────────────────────────┐
#        │ 100 images: split="test" (untouched)    │
#        │ 100 images: split="train" or "val"      │ ← 50 new ones assigned
#        │ 800 images: split=NULL (still waiting)  │
#        └─────────────────────────────────────────┘

# DAY 7: Final state after 500 labels
#        ┌─────────────────────────────────────────┐
#        │ 100 images: split="test" (STILL untouched) │
#        │ 500 images: split="train" (~400) or "val" (~100) │
#        │ 400 images: split=NULL (never labeled)  │
#        └─────────────────────────────────────────┘

def _no_holdout(rows):
    "Safety check: crash if hold-out leaked into training data"
    assert all(r.get("split") != "test" for r in rows), "Hold-out leak!"

# =============================================================================
# PUBLIC API
# =============================================================================

def get_db(path=None):
    "Get database connection, init schema, assign hold-outs"
    db = Database(str(ifnone(path, DB_PATH)))
    db.execute("PRAGMA journal_mode=WAL")
    # This is a cool design pattern; WAL (Write-Ahead Logging) is a SQLite feature that allows for concurrent reads and writes.
    # When you say "df['labels'].update(5, {'confidence_biopsy': 0.92})" that change is immediate, it's written in WAL file, this is a commit
    # When we hit a checkpoint, the SQLite engine merges the WAL file into the main .db and flushes the WAL file
# ┌─────────────────────────────────────────────────────────────┐
# │                        READ FLOW                            │
# │                                                             │
# │  Query: SELECT row 5                                        │
# │           ↓                                                 │
# │  Check WAL index (-shm): "row 5 modified?"                 │
# │           ↓                                                 │
# │     ┌─────┴─────┐                                          │
# │    YES          NO                                          │
# │     ↓            ↓                                          │
# │  Find latest   Read from                                    │
# │  TX for row 5  main .db                                     │
# │     ↓                                                       │
# │  Return that                                                │
# │  state (UPDATE                                              │
# │  /DELETE/etc)                                               │
# └─────────────────────────────────────────────────────────────┘
    _init_db(db)
    return db

# image enters the database ; essentially all NULL values
def insert_image(db, filename, study_id, thumbnail_path, frame_number=0):
    "Insert new image record. Skips duplicates."
    if image_exists(db, filename): logger.warning(f"Duplicate skipped: {filename}"); return

    # Holdout decision is per-image, based on hash
    # Let's expand on this further: _hash_frac combines seed 42 with filename, 
    # converts string to bytes using SHA256, produced a 64-char hex string like 'a2f23...' 
    # takes the first 8 hex chars, converts hex to int, divides by max 8-hex-digit. 
    # WHY uniform 0-1? SHA256 is designed to output evenly across all possible values. 
    # no input pattern produce clustered outputs. so when we divide by max value, 
    # we get numbers spread evenly between 0 and 1. it is a uniform distribution. 
    # SHA256 looks random (uniform spread) but it's completely reproducible. 
    # Same inp, same outs. you get statistical properties of randomness without actual randomness.
    split = "test" if _hash_frac(filename) < HOLDOUT_PCT else None

    db["labels"].insert({
        "filename": filename, "study_id": study_id, "thumbnail_path": thumbnail_path,
        "frame_number": frame_number, "has_biopsy_tool": None, "has_mag_view": None,
        "labeled_at": None, "confidence_biopsy": None, "confidence_mag": None,
        "predicted_at": None, "split": split,
    }, ignore=True)
    logger.info(f"Inserted {filename}")

def get_unlabeled(db, limit=1, exploration_rate=EXPLORATION_RATE, rng=None):
    "Active learning: balance uncertain, confident, and random samples"
    rng = rng or random
    where = ("has_biopsy_tool IS NULL AND has_mag_view IS NULL "
        "AND (split IS NULL OR split IN ('train', 'val'))")
    
    # DEBUG
    count = db.execute(f"SELECT COUNT(*) FROM labels WHERE {where}").fetchone()[0]
    print(f"DEBUG get_unlabeled: {count} candidates")
    
    r = rng.random()
    if r < exploration_rate:
        # 10% exploration: random
        rows = list(db["labels"].rows_where(where))
        if not rows: return []
        rng.shuffle(rows)
        return rows[:limit]
    elif r < exploration_rate + 0.45:
        # 45% uncertain: closest to 0.5
        return list(db["labels"].rows_where(
            where, order_by="ABS(COALESCE(confidence_biopsy, 0.5) - 0.5) ASC, RANDOM()", limit=limit))
    else:
        # 45% confident: closest to 0.0 or 1.0
        return list(db["labels"].rows_where(
            where, order_by="ABS(COALESCE(confidence_biopsy, 0.5) - 0.5) DESC, RANDOM()", limit=limit))

def set_labels(db, image_id, biopsy, mag):
    "Set both labels, assign split if needed"
    db["labels"].update(image_id, {"has_biopsy_tool": biopsy, "has_mag_view": mag,
        "labeled_at": datetime.now().isoformat(),})
    _ensure_split(db, image_id)

def get_labeled(db, split=None):
    "Get fully labeled images. Default: train+val only."
    if split and split not in ("train", "val", "test"): raise ValueError(f"Invalid split: {split}")

    base = "has_biopsy_tool IS NOT NULL AND has_mag_view IS NOT NULL"
    if split: rows = list(db["label"].rows_where(f"{base} AND split = ?", [split]))
    else:     rows = list(db["labels"].rows_where(f"{base} AND split IN ('train', 'val')"))

    if split != "test": _no_holdout(rows)
    return rows

def get_train_val(db):
    "Return (train_rows, val_rows) for training"
    train, val = get_labeled(db, "train"), get_labeled(db, "val")
    _no_holdout(train + val)
    return train, val

def update_predictions(db, preds):
    "Bulk update confidences for unlabeled images. Skips hold-out and already-labeled."
    now = datetime.now().isoformat()
    for p in preds:
        row = get_image_by_id(db, p["id"])
        if not row: continue
        if row.get("split") == "test": continue
        # Only update if unlabeled - skip train/val
        if row.get("has_biopsy_tool") is not None: continue
        db["labels"].update(p["id"], {
            "confidence_biopsy": p.get("confidence_biopsy"),
            "confidence_mag": p.get("confidence_mag"),
            "predicted_at": now,
        })
    logger.info(f"Updated {len(preds)} predictions")

def get_stats(db):
    "Labeling statistics"
    q = lambda sql: db.execute(sql).fetchone()[0]
    return {
        "total": db["labels"].count,
        "biopsy_labeled": q("SELECT COUNT(*) FROM labels WHERE has_biopsy_tool IS NOT NULL"),
        "biopsy_yes":     q("SELECT COUNT(*) FROM labels WHERE has_biopsy_tool = 1"),
        "mag_labeled":    q("SELECT COUNT(*) FROM labels WHERE has_mag_view IS NOT NULL"),
        "mag_yes":        q("SELECT COUNT(*) FROM labels WHERE has_mag_view = 1"),
        "fully_labeled":  q("SELECT COUNT(*) FROM labels WHERE has_biopsy_tool IS NOT NULL AND has_mag_view IS NOT NULL"),
    }

def get_image_by_id(db, image_id):
    try: return db["labels"].get(image_id)
    except: return None

def image_exists(db, filename): return db.execute("SELECT 1 FROM labels wHERE filename = ? LIMIT 1", [filename]).fetchone() is not None

# IMPORTANT: Predictions are run on **unlabeled** images, not on va/train
# ┌─────────────────────────────────────────────────────────────────┐
# │ ITERATION 1                                                      │
# ├─────────────────────────────────────────────────────────────────┤
# │                                                                  │
# │  1000 images total                                               │
# │  ├── 100 hold-out (split="test", untouched forever)             │
# │  └── 900 available (split=NULL, unlabeled)                       │
# │                                                                  │
# │  You label 50 random images                                      │
# │  ├── ~40 become split="train"                                    │
# │  └── ~10 become split="val"                                      │
# │                                                                  │
# │  Train model on 40 train images                                  │
# │  Validate on 10 val images → 67% accuracy                        │
# │                                                                  │
# │  Run predictions on 850 UNLABELED images ← THIS IS KEY           │
# │  Each gets confidence_biopsy, confidence_mag scores              │
# │                                                                  │
# └─────────────────────────────────────────────────────────────────┘

# ┌─────────────────────────────────────────────────────────────────┐
# │ ITERATION 2                                                      │
# ├─────────────────────────────────────────────────────────────────┤
# │                                                                  │
# │  get_unlabeled() looks at 850 unlabeled images                   │
# │  Returns images where model is:                                  │
# │  ├── 10% random (exploration)                                    │
# │  ├── 45% uncertain (confidence ≈ 0.5)                            │
# │  └── 45% confident (confidence ≈ 0.0 or ≈ 1.0) ← CATCHES ERRORS │
# │                                                                  │
# │  You label 50 more images                                        │
# │  These 50 get assigned to train/val via hash                     │
# │                                                                  │
# │  Now you have:                                                   │
# │  ├── 100 labeled (80 train, 20 val)                              │
# │  └── 800 unlabeled (with predictions)                            │
# │                                                                  │
# │  Train model on 80 train images                                  │
# │  Validate on 20 val images → 78% accuracy                        │
# │                                                                  │
# │  Run predictions on 800 UNLABELED images                         │
# │                                                                  │
# └─────────────────────────────────────────────────────────────────┘