from pathlib import Path

import pytest
import pydicom

from lib.config import HOLDOUT_PCT
from lib.db import get_db, insert_image, set_labels, update_predictions, assign_holdout_splits
from lib.dicom_utils import find_dicoms, get_frame_count, create_thumbnail


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "samp_ds"


def _build_db_from_samples(tmp_path: Path):
    if not SAMPLE_DIR.exists():
        pytest.fail(f"Sample DICOM dir not found: {SAMPLE_DIR}")
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "labels.db"
    thumb_dir = tmp_path / "thumbs"
    db = get_db(db_path)
    inserted = []

    for dcm_path, study_id in find_dicoms(SAMPLE_DIR):
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            num_frames = get_frame_count(ds)
        except Exception:
            continue

        for frame_idx in range(num_frames):
            filename = f"{study_id}/{dcm_path.stem}"
            if num_frames > 1:
                filename = f"{filename}_frame{frame_idx:04d}"
            thumb_name = filename.replace("/", "_") + ".png"
            thumb_path = thumb_dir / thumb_name
            if create_thumbnail(dcm_path, thumb_path, frame_idx) is None:
                continue
            insert_image(db, filename, study_id, str(thumb_path), frame_idx)
            inserted.append(filename)

    if not inserted:
        pytest.fail("No thumbnails created from samp_ds")
    assign_holdout_splits(db)
    return db


def test_integration_pipeline_splits_and_predictions(tmp_path):
    db = _build_db_from_samples(tmp_path)

    holdout_rows = list(db["labels"].rows_where("split = 'test'"))
    total = db["labels"].count
    assert abs((len(holdout_rows) / total) - HOLDOUT_PCT) <= 0.01

    candidates = list(db["labels"].rows_where("split IS NULL", order_by="id", limit=20))
    if len(candidates) < 20:
        pytest.fail("Not enough non-holdout images for labeling")
    split_before = {}
    for idx, row in enumerate(candidates):
        set_labels(db, row["id"], idx % 2, (idx + 1) % 2)
        updated = db["labels"].get(row["id"])
        assert updated["split"] in ("train", "val")
        split_before[row["id"]] = updated["split"]

    preds = []
    for row in db["labels"].rows_where("1=1"):
        preds.append({
            "id": row["id"],
            "confidence_biopsy": 0.25,
            "confidence_mag": 0.75,
        })
    update_predictions(db, preds)

    for row in db["labels"].rows_where("1=1"):
        updated = db["labels"].get(row["id"])
        if updated["split"] == "test":
            assert updated["predicted_at"] is None
        else:
            assert updated["predicted_at"] is not None

    for idx, row in enumerate(candidates):
        set_labels(db, row["id"], (idx + 1) % 2, idx % 2)
        relabeled = db["labels"].get(row["id"])
        assert relabeled["split"] == split_before[row["id"]]
