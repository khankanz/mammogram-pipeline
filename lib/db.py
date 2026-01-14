"""SQLite database helpers using sqlite-utils."""

from sqlite_utils import Database
from pathlib import Path
from datetime import datetime
from typing import Optional
from .config import DB_PATH


def get_db(path: Optional[Path] = None) -> Database:
    """Get database connection, creating schema if needed."""
    db_path = path or DB_PATH
    db = Database(str(db_path))

    # Enable WAL mode for better concurrency
    db.execute("PRAGMA journal_mode=WAL")

    if "labels" not in db.table_names():
        db["labels"].create({
            "id": int,
            "filename": str,
            "study_id": str,  # Folder/study name
            "thumbnail_path": str,
            "frame_number": int,  # For multi-frame DICOMs
            "has_biopsy_tool": int,
            "has_mag_view": int,
            "labeled_at": str,
            "confidence_biopsy": float,
            "confidence_mag": float,
            "predicted_at": str,
        }, pk="id")
        db["labels"].create_index(["filename"], unique=True)
        db["labels"].create_index(["study_id"])
        db["labels"].create_index(["has_biopsy_tool", "has_mag_view"])
        db["labels"].create_index(["confidence_biopsy"])
        db["labels"].create_index(["confidence_mag"])

    return db


def insert_image(db: Database, filename: str, study_id: str,
                 thumbnail_path: str, frame_number: int = 0) -> None:
    """Insert a new image record (skip if exists)."""
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
    }, ignore=True)


def get_unlabeled(db: Database, limit: int = 1) -> list:
    """Get images needing labels, prioritize low confidence predictions."""
    return list(db["labels"].rows_where(
        "has_biopsy_tool IS NULL AND has_mag_view IS NULL",
        order_by="confidence_biopsy ASC NULLS FIRST",
        limit=limit
    ))


def get_partially_labeled(db: Database, limit: int = 1) -> list:
    """Get images with only one label set."""
    return list(db["labels"].rows_where(
        "(has_biopsy_tool IS NULL) != (has_mag_view IS NULL)",
        limit=limit
    ))


def set_label(db: Database, image_id: int, field: str, value: int) -> None:
    """Set a label for an image."""
    if field not in ("biopsy", "mag"):
        raise ValueError(f"Invalid field: {field}")

    col = "has_biopsy_tool" if field == "biopsy" else "has_mag_view"
    db["labels"].update(image_id, {
        col: value,
        "labeled_at": datetime.now().isoformat(),
    })


def clear_label(db: Database, image_id: int, field: str) -> None:
    """Clear a label (for undo)."""
    if field not in ("biopsy", "mag"):
        raise ValueError(f"Invalid field: {field}")

    col = "has_biopsy_tool" if field == "biopsy" else "has_mag_view"
    db["labels"].update(image_id, {col: None})


def get_labeled_images(db: Database) -> list:
    """Get all fully labeled images for training."""
    return list(db["labels"].rows_where(
        "has_biopsy_tool IS NOT NULL AND has_mag_view IS NOT NULL"
    ))


def update_predictions(db: Database, predictions: list[dict]) -> None:
    """Bulk update prediction confidences."""
    now = datetime.now().isoformat()
    for pred in predictions:
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
