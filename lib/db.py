"""SQLite database helpers using sqlite-utils."""

from sqlite_utils import Database
from pathlib import Path
from datetime import datetime
from typing import Optional
import hashlib
import logging
import random
from .config import DB_PATH, HOLDOUT_PCT, EXPLORATION_RATE, RANDOM_SEED, VALID_PCT


from fastcore.basics import ifnone, store_attr
from functools import lru_cache

logger = logging.getLogger(__name__)

# Schema definition - all in one place, easy to scan
_label_schema = dict(
    id=int, filename=str, study_id=str, thumbnail_path=str, frame_number=int,
    has_biopsy_tool=int, has_mag_view=int, labeled_at=str,
    confidence_biopsy=float, confidence_mag=float, predicted_at=str,
    split=str)

_label_indexes = [(["filename"], {"unique": True}),
    (["study_id"], {}),
    (["has_biopsy_tool", "has_mag_view"], {}),
    (["confidence_biopsy"], {}),
    (["confidence_mag"], {}),
    (["split"], {}),
]

def _init_labels_table(db):
    "Create labels table with indexes if not exists"
    if "labels" in db.table_names(): return
    db["labels"].create(_label_schema, pk="id")
    for cols,kw in _label_indexes: db["labels"].create_index(cols,**kw)

def _ensure_labels_schema(db):
    "Ensure labels table has expected columns and indexes"
    if "labels" not in db.table_names():
        return
    table = db["labels"]
    if "split" not in table.columns_dict:
        table.add_column("split", str)
    for cols, kw in _label_indexes:
        table.create_index(cols, **kw, if_not_exists=True)

def _hash_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF

def _train_val_split(filename: str) -> str:
    frac = _hash_fraction(f"trainval:{filename}", RANDOM_SEED)
    return "val" if frac < VALID_PCT else "train"

def assign_holdout_splits(db: Database) -> int:
    """Assign deterministic hold-out splits for unlabeled, unassigned rows."""
    rows = list(db["labels"].rows_where(
        "split IS NULL AND has_biopsy_tool IS NULL AND has_mag_view IS NULL"
    ))
    if not rows:
        return 0
    total = len(rows)
    target = int(round(total * HOLDOUT_PCT))
    if target <= 0:
        return 0
    ranked = sorted(
        rows,
        key=lambda r: (_hash_fraction(r["filename"], RANDOM_SEED), r["filename"])
    )
    for row in ranked[:target]:
        db["labels"].update(row["id"], {"split": "test"})
    logger.info("Assigned hold-out split to %d images", target)
    return target

def _backfill_splits(db):
    "Assign hold-out/test and train/val splits where missing."
    if "labels" not in db.table_names():
        return
    table = db["labels"]
    assign_holdout_splits(db)
    updated_trainval = 0
    for row in table.rows_where(
        "split IS NULL AND has_biopsy_tool IS NOT NULL AND has_mag_view IS NOT NULL"
    ):
        split = _train_val_split(row["filename"])
        table.update(row["id"], {"split": split})
        updated_trainval += 1
    if updated_trainval:
        logger.info("Assigned train/val split to %d labeled images", updated_trainval)

def _ensure_split_for_image(db: Database, image_id: int) -> None:
    row = db["labels"].get(image_id)
    current_split = row.get("split")
    if current_split == "test":
        return
    if current_split in ("train", "val"):
        return
    split = _train_val_split(row["filename"])
    db["labels"].update(image_id, {"split": split})

def _assert_no_holdout(rows: list[dict]) -> None:
    assert all(r.get("split") != "test" for r in rows), "Hold-out image in training data"

def get_db(path=None):
    "Get database connection, creating schema if needed"
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
    _init_labels_table(db)
    _ensure_labels_schema(db)
    _backfill_splits(db)
    return db

def insert_image(db: Database, filename: str, study_id: str,
                 thumbnail_path: str, frame_number: int = 0) -> None:
    """Insert a new image record (skip if exists)."""
    if image_exists(db, filename):
        logger.warning("Duplicate image insert skipped: %s", filename)
        return
    db["labels"].insert({
        "filename": filename,
        "study_id": study_id,
        "thumbnail_path": thumbnail_path,
        "frame_number": frame_number,
        "has_biopsy_tool": None,
        "has_mag_view": None,
        "labeled_at": None,
        "confidence_biopsy": None,
        "confidence_mag": None,
        "predicted_at": None,
        "split": None,
    }, ignore=True)


def get_unlabeled(db: Database, limit: int = 1,
                  exploration_rate: float = EXPLORATION_RATE,
                  rng: Optional[random.Random] = None) -> list:
    """Get images needing labels, prioritize low confidence predictions."""
    # This is the heat of active learning. Order is: 1) images with no predictions yet NULL; brand new images
    # 2) images where model is uncertain; 0.45, 0.5 etc. #3 images where model is confident; 0.02 and 0.98
    # In this order, we're attempting to find no predictions exist so those randomly come up, then after training the uncertrain predictions come first
    rng = rng or random
    where_clause = ("has_biopsy_tool IS NULL AND has_mag_view IS NULL "
                    "AND (split IS NULL OR split IN ('train', 'val'))")
    if rng.random() < exploration_rate:
        rows = list(db["labels"].rows_where(where_clause, order_by="id"))
        if not rows:
            return []
        picked = []
        for _ in range(min(limit, len(rows))):
            idx = int(rng.random() * len(rows))
            picked.append(rows.pop(idx))
        return picked
    return list(db["labels"].rows_where(
        where_clause,
        order_by="confidence_biopsy ASC NULLS FIRST",
        limit=limit
    ))


def get_partially_labeled(db: Database, limit: int = 1) -> list:
    """Get images with only one label set."""
    return list(db["labels"].rows_where(
        "(has_biopsy_tool IS NULL) != (has_mag_view IS NULL) "
        "AND (split IS NULL OR split IN ('train', 'val'))",
        limit=limit
    ))
    # we're trying to be slick with != boolean to say find me instances were we didn't label has_biopsy_tool or has_mag_view; both should have a label associated; either 0 or 1
    # we might want to consider changing boolean to make it more clear to reader


def set_label(db: Database, image_id: int, field: str, value: int) -> None:
    """Set a label for an image."""
    if field not in ("biopsy", "mag"):
        raise ValueError(f"Invalid field: {field}")

    col = "has_biopsy_tool" if field == "biopsy" else "has_mag_view"
    db["labels"].update(image_id, {
        col: value,
        "labeled_at": datetime.now().isoformat(),
    })
    _ensure_split_for_image(db, image_id)

def set_labels(db: Database, image_id: int, biopsy: int, mag: int) -> None:
    """Set both labels for an image, assign split if needed."""
    db["labels"].update(image_id, {
        "has_biopsy_tool": biopsy,
        "has_mag_view": mag,
        "labeled_at": datetime.now().isoformat(),
    })
    _ensure_split_for_image(db, image_id)


def clear_label(db: Database, image_id: int, field: str) -> None:
    """Clear a label (for undo)."""
    # This is used in our undo feature
    if field not in ("biopsy", "mag"):
        raise ValueError(f"Invalid field: {field}")

    col = "has_biopsy_tool" if field == "biopsy" else "has_mag_view"
    db["labels"].update(image_id, {col: None})


def get_labeled_images(db: Database, split: Optional[str] = None) -> list:
    """Get fully labeled images for training or evaluation."""
    if split and split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split: {split}")
    where = "has_biopsy_tool IS NOT NULL AND has_mag_view IS NOT NULL"
    if split:
        where = f"{where} AND split = ?"
        rows = list(db["labels"].rows_where(where, [split]))
    else:
        where = f"{where} AND split IN ('train', 'val')"
        rows = list(db["labels"].rows_where(where))
    if split in (None, "train", "val"):
        _assert_no_holdout(rows)
    return rows

def get_train_val_split(db: Database) -> tuple[list, list]:
    """Return labeled train and val splits."""
    train_rows = get_labeled_images(db, split="train")
    val_rows = get_labeled_images(db, split="val")
    _assert_no_holdout(train_rows + val_rows)
    return train_rows, val_rows
    # this is our Training Data Query; only return images where both labels are set; training needs complete data

def update_predictions(db: Database, predictions: list[dict]) -> None:
    """Bulk update prediction confidences."""
    now = datetime.now().isoformat()
    for pred in predictions:
        row = get_image_by_id(db, pred["id"])
        if not row:
            logger.warning("Prediction skipped; missing image id %s", pred.get("id"))
            continue
        if row.get("split") == "test":
            logger.warning("Prediction skipped for hold-out image id %s", pred.get("id"))
            continue
        db["labels"].update(pred["id"], {
            "confidence_biopsy": pred.get("confidence_biopsy"),
            "confidence_mag": pred.get("confidence_mag"),
            "predicted_at": now,
        })


def get_stats(db: Database) -> dict:
    """Get labeling statistics."""
    total = db["labels"].count

    biopsy_labeled = db.execute(
        "SELECT COUNT(*) FROM labels WHERE has_biopsy_tool IS NOT NULL"
    ).fetchone()[0]
    biopsy_yes = db.execute(
        "SELECT COUNT(*) FROM labels WHERE has_biopsy_tool = 1"
    ).fetchone()[0]

    mag_labeled = db.execute(
        "SELECT COUNT(*) FROM labels WHERE has_mag_view IS NOT NULL"
    ).fetchone()[0]
    mag_yes = db.execute(
        "SELECT COUNT(*) FROM labels WHERE has_mag_view = 1"
    ).fetchone()[0]

    fully_labeled = db.execute(
        "SELECT COUNT(*) FROM labels WHERE has_biopsy_tool IS NOT NULL AND has_mag_view IS NOT NULL"
    ).fetchone()[0]

    return {
        "total": total,
        "biopsy_labeled": biopsy_labeled,
        "biopsy_yes": biopsy_yes,
        "mag_labeled": mag_labeled,
        "mag_yes": mag_yes,
        "fully_labeled": fully_labeled,
    }


def get_image_by_id(db: Database, image_id: int) -> Optional[dict]:
    """Get a single image by ID."""
    try:
        return db["labels"].get(image_id)
    except Exception:
        return None


def image_exists(db: Database, filename: str) -> bool:
    """Check if an image already exists in database."""
    return db.execute(
        "SELECT 1 FROM labels WHERE filename = ? LIMIT 1", [filename]
    ).fetchone() is not None
