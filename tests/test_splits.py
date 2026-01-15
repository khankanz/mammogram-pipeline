import random
from pathlib import Path

import pytest
import pydicom

from lib.config import HOLDOUT_PCT
from lib.db import (
    get_db,
    insert_image,
    get_unlabeled,
    update_predictions,
    set_labels,
    assign_holdout_splits,
)
from lib.dicom_utils import find_dicoms, get_frame_count, create_thumbnail


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "samp_ds"


def _build_db_from_samples(tmp_path: Path, reverse: bool = False):
    if not SAMPLE_DIR.exists():
        pytest.fail(f"Sample DICOM dir not found: {SAMPLE_DIR}")
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "labels.db"
    thumb_dir = tmp_path / "thumbs"
    db = get_db(db_path)
    inserted = []

    dicoms = list(find_dicoms(SAMPLE_DIR))
    if reverse:
        dicoms = list(reversed(dicoms))

    for dcm_path, study_id in dicoms:
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
    return db, inserted


def test_holdout_deterministic_assignment(tmp_path):
    db1, _ = _build_db_from_samples(tmp_path / "db1")
    db2, _ = _build_db_from_samples(tmp_path / "db2", reverse=True)

    holdout1 = {row["filename"] for row in db1["labels"].rows_where("split = 'test'")}
    holdout2 = {row["filename"] for row in db2["labels"].rows_where("split = 'test'")}

    total = db1["labels"].count
    assert holdout1 == holdout2
    assert abs((len(holdout1) / total) - HOLDOUT_PCT) <= 0.01


def test_get_unlabeled_exploration_and_holdout_exclusion(tmp_path):
    db, _ = _build_db_from_samples(tmp_path)
    rows = list(db["labels"].rows_where("split IS NULL", order_by="id"))
    if len(rows) < 10:
        pytest.fail("Not enough non-holdout rows for exploration test")

    target_id = rows[0]["id"]
    for row in rows:
        db["labels"].update(row["id"], {"confidence_biopsy": 0.9})
    db["labels"].update(target_id, {"confidence_biopsy": 0.1})

    rng = random.Random(12345)
    exploration_hits = 0
    for _ in range(100):
        result = get_unlabeled(db, limit=1, exploration_rate=0.1, rng=rng)
        assert result
        assert result[0]["split"] != "test"
        if result[0]["id"] != target_id:
            exploration_hits += 1

    assert 5 <= exploration_hits <= 20


def test_update_predictions_skips_holdout(tmp_path):
    db, _ = _build_db_from_samples(tmp_path)
    rows = list(db["labels"].rows_where("1=1"))
    holdout = [r for r in rows if r.get("split") == "test"]
    non_holdout = [r for r in rows if r.get("split") != "test"]
    if not holdout or not non_holdout:
        pytest.fail("Need both holdout and non-holdout rows")
    holdout_row = holdout[0]
    normal_row = non_holdout[0]

    update_predictions(db, [
        {"id": holdout_row["id"], "confidence_biopsy": 0.1, "confidence_mag": 0.2},
        {"id": normal_row["id"], "confidence_biopsy": 0.8, "confidence_mag": 0.9},
    ])

    holdout_row = db["labels"].get(holdout_row["id"])
    normal_row = db["labels"].get(normal_row["id"])
    assert holdout_row["predicted_at"] is None
    assert normal_row["predicted_at"] is not None


def test_split_assignment_is_permanent(tmp_path):
    db, _ = _build_db_from_samples(tmp_path)
    row = next(iter(db["labels"].rows_where("split IS NULL")), None)
    if not row:
        pytest.fail("No non-holdout row found for split assignment")

    set_labels(db, row["id"], 1, 0)
    assigned = db["labels"].get(row["id"])
    assert assigned["split"] in ("train", "val")

    set_labels(db, row["id"], 0, 1)
    relabeled = db["labels"].get(row["id"])
    assert relabeled["split"] == assigned["split"]


def test_duplicate_insert_logs_warning(tmp_path, caplog):
    db, _ = _build_db_from_samples(tmp_path)
    row = next(iter(db["labels"].rows_where("1=1")), None)
    if not row:
        pytest.fail("No rows inserted from samp_ds")

    with caplog.at_level("WARNING"):
        insert_image(db, row["filename"], row["study_id"], row["thumbnail_path"], row["frame_number"])

    assert any("Duplicate image insert skipped" in record.message for record in caplog.records)
