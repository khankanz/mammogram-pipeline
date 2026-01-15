#!/usr/bin/env python3
"""Preprocess DICOMs: convert to normalized 224x224 PNGs and populate database.

Processes all frames from multi-frame DICOMs.
"""

import argparse
import sys
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import logging

# Add parent dir to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import (
    DICOM_DIR, THUMBNAIL_DIR, DB_PATH, TEST_DICOM_DIR,
    NUM_WORKERS, PROGRESS_INTERVAL, IMAGE_SIZE, ensure_dirs
)
from lib.db import get_db, insert_image, image_exists, assign_holdout_splits
from lib.dicom_utils import find_dicoms, get_frame_count, create_thumbnail

logger = logging.getLogger(__name__)


def process_single_dicom(dicom_path: Path, study_id: str,
                         output_dir: Path, db_path: Path) -> list[dict]:
    """Process a single DICOM file, handling multi-frame.

    Returns list of successfully processed frames as dicts.
    """
    results = []

    try:
        import pydicom
        ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=False)
        num_frames = get_frame_count(ds)
    except Exception as e:
        print(f"Warning: Cannot read {dicom_path}: {e}")
        return results

    for frame_idx in range(num_frames):
        # Generate unique filename for this frame
        if num_frames > 1:
            filename = f"{study_id}/{dicom_path.stem}_frame{frame_idx:04d}"
        else:
            filename = f"{study_id}/{dicom_path.stem}"

        # Thumbnail path (relative to project)
        thumb_name = filename.replace("/", "_") + ".png"
        thumb_path = output_dir / thumb_name

        # Skip if already processed
        if thumb_path.exists():
            results.append({
                "filename": filename,
                "study_id": study_id,
                "thumbnail_path": str(thumb_path),
                "frame_number": frame_idx,
            })
            continue

        # Create thumbnail
        if create_thumbnail(dicom_path, thumb_path, frame_idx, IMAGE_SIZE):
            results.append({
                "filename": filename,
                "study_id": study_id,
                "thumbnail_path": str(thumb_path),
                "frame_number": frame_idx,
            })

    return results


def preprocess_directory(dicom_dir: Path, output_dir: Path, db_path: Path,
                         n_jobs: int = NUM_WORKERS) -> int:
    """Process all DICOMs in directory tree.

    Args:
        dicom_dir: Root directory containing DICOM files
        output_dir: Directory for output thumbnails
        db_path: Path to SQLite database
        n_jobs: Number of parallel workers

    Returns:
        Number of images processed
    """
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all DICOM files
    print(f"Scanning {dicom_dir} for DICOM files...")
    dicom_files = list(find_dicoms(dicom_dir))
    print(f"Found {len(dicom_files)} DICOM files")

    if not dicom_files:
        print("No DICOM files found. Exiting.")
        return 0

    # Process in parallel
    print(f"Processing with {n_jobs} workers...")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_dicom)(dcm_path, study_id, output_dir, db_path)
        for dcm_path, study_id in dicom_files
    )

    # Flatten results
    all_frames = [frame for file_frames in results for frame in file_frames]
    print(f"Processed {len(all_frames)} total frames")

    # Insert into database
    print("Updating database...")
    db = get_db(db_path)

    inserted = 0
    for frame in tqdm(all_frames, desc="Inserting to DB"):
        if not image_exists(db, frame["filename"]):
            insert_image(
                db,
                filename=frame["filename"],
                study_id=frame["study_id"],
                thumbnail_path=frame["thumbnail_path"],
                frame_number=frame["frame_number"],
            )
            inserted += 1
        else:
            logger.warning("Skipping duplicate image: %s", frame["filename"])

    print(f"Inserted {inserted} new records into database")
    print(f"Total images in database: {db['labels'].count}")
    assign_holdout_splits(db)

    return len(all_frames)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DICOM files to thumbnails"
    )
    parser.add_argument(
        "--dicom-dir", type=Path, default=DICOM_DIR,
        help=f"Directory containing DICOM files (default: {DICOM_DIR})"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=THUMBNAIL_DIR,
        help=f"Output directory for thumbnails (default: {THUMBNAIL_DIR})"
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help=f"SQLite database path (default: {DB_PATH})"
    )
    parser.add_argument(
        "--workers", type=int, default=NUM_WORKERS,
        help=f"Number of parallel workers (default: {NUM_WORKERS})"
    )
    parser.add_argument(
        "--test", action="store_true",
        help=f"Use test directory ({TEST_DICOM_DIR}) instead"
    )

    args = parser.parse_args()

    dicom_dir = TEST_DICOM_DIR if args.test else args.dicom_dir

    if not dicom_dir.exists():
        print(f"Error: DICOM directory does not exist: {dicom_dir}")
        print("Please create the directory and add DICOM files.")
        sys.exit(1)

    count = preprocess_directory(
        dicom_dir=dicom_dir,
        output_dir=args.output_dir,
        db_path=args.db,
        n_jobs=args.workers,
    )

    print(f"\nPreprocessing complete. {count} frames ready for labeling.")
    print(f"Run: python scripts/02_label_server.py")


if __name__ == "__main__":
    main()
