#!/usr/bin/env python3
"""Preprocess DICOMs: convert to normalized thumbnails and populate database."""

import sys, time, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastcore.all import *
from fastcore.script import call_parse
from fastprogress.fastprogress import progress_bar

import pydicom
from lib.config import (DICOM_DIR, THUMBNAIL_DIR, DB_PATH, TEST_DICOM_DIR,
                        NUM_WORKERS, IMAGE_SIZE, ensure_dirs)
from lib.db import get_db, insert_image, image_exists
from lib.dicom_utils import find_dicoms, get_frame_count, create_thumbnail
import warnings

logger = logging.getLogger(__name__)

# =============================================================================
# TIMING UTILITIES
# =============================================================================

class Timer:
    "Simple timer that tracks elapsed time and throughput."
    def __init__(self, name=''):
        self.name, self.start, self.n = name, time.perf_counter(), 0
    
    def __repr__(self):
        elapsed = time.perf_counter() - self.start
        rate = self.n / elapsed if elapsed > 0 else 0
        return f"{self.name}: {self.n:,} items in {elapsed:.1f}s ({rate:.1f}/sec)"
    
    def add(self, n=1): self.n += n; return self

# =============================================================================
# CORE PROCESSING
# =============================================================================

def _frame_filename(study_id, stem, frame_idx, num_frames):
    "Generate unique filenames for a frame."
    base = f"{study_id}/{stem}"
    return f"{base}_frame{frame_idx:04d}" if num_frames > 1 else base

def _process_frame(dcm_path, study_id, frame_idx, num_frames, out_dir):
    "Process a single frame, return dict or None"
    filename = _frame_filename(study_id, dcm_path.stem, frame_idx, num_frames)
    thumb_name = filename.replace("/","_") + ".png"
    thumb_path = out_dir / thumb_name

    # Idempotent: skip if exists
    if thumb_path.exists(): return dict(filename=filename, study_id=study_id, thumbnail_path=str(thumb_path))
    # Create thumbnail
    if create_thumbnail(dcm_path, thumb_path, frame_idx, IMAGE_SIZE): return dict(filename=filename, study_id=study_id, 
        thumbnail_path=str(thumb_path), frame_number=frame_idx)
    return None

def process_dicom(dcm_path, study_id,out_dir):
    "Process a single DICOM file, return list of frame dicts"
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=False) # Why? Because we need the pixel data to generate thumbnails
        num_frames = get_frame_count(ds)
    except Exception as e:
        logger.warning(f"Cannot read {dcm_path}: {e}")
        return []
    
    results = []
    for i in range(num_frames):
        r = _process_frame(dcm_path, study_id, i, num_frames, out_dir)
        if r: results.append(r)
    return results

def _process_wrapper(item, out_dir):
    "Wrapper for parallel - unpacks tuple"
    warnings.simplefilter("ignore")
    dcm_path, study_id = item
    return process_dicom(dcm_path, study_id, out_dir)

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def preprocess(dicom_dir, output_dir, db_path, n_workers=NUM_WORKERS):
    "Process all DICOMs in directory tree"
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discovery
    t_discover = Timer("Discover")
    print(f"Scanning {dicom_dir} for DICOMs files...")
    dicoms = find_dicoms(dicom_dir)
    t_discover.add(len(dicoms))
    
    if not dicoms: print("No DICOMs found. Exiting."); return 0

    # process in parallel with progress bar
    t_process = Timer("Processing")
    print(f"\nProcessing with {n_workers} workers...")

    # fastcore.parallel with progress=True gives us a progress bar
    results = parallel(partial(_process_wrapper, out_dir=output_dir),
        dicoms, n_workers=n_workers, progress=True, threadpool=False) # Uses processes for CPU-bound DICOM work
    
    # Flatten results
    all_frames = L(results).concat()
    t_process.add(len(all_frames))
    print(f"    {t_process}")

    # Database insertion
    t_db = Timer("DB Insert")
    print(f"\nUpdating database....")
    db = get_db(db_path)

    inserted = 0
    for frame in progress_bar(all_frames, leave=False):
        if not image_exists(db, frame["filename"]):
            insert_image(db, frame["filename"], frame["study_id"], 
                frame["thumbnail_path"], frame["frame_number"])
            inserted += 1
    t_db.add(inserted)

    print(f"  {t_db}")
    print(f"  Total in database: {db['labels'].count:,}")
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"  {t_discover}")
    print(f"  {t_process}")
    print(f"  {t_db}")
    print(f"{'='*50}")
    
    return len(all_frames)

# =============================================================================
# CLI
# =============================================================================

# This is a decorator that turns a regular Python function into a CLI script automatically. It reads your function's
# 1) parameter names -> become CLI arg names 2) Type annotations -> become argument types 3) Default values -> determine if args is required or optional
# 4) Comments after params -> become --help text
# This elminates ~20 ines of argparse boilerplate by introspecting your function signature
@call_parse
def main(
    dicom_dir:Path=DICOM_DIR,   # Directory containing DICOM files
    output_dir:Path=THUMBNAIL_DIR, # Output directory for thumbnails
    db:Path=DB_PATH,            # SQLite database path
    workers:int=NUM_WORKERS,    # Number of parallel workers
    test:bool_arg=False         # Use test directory instead
):
    "Preprocess DICOM files to thumbnails"
    dicom_dir = TEST_DICOM_DIR if test else dicom_dir
    
    if not dicom_dir.exists():
        print(f"Error: DICOM directory does not exist: {dicom_dir}")
        print("Please create the directory and add DICOM files.")
        sys.exit(1)
    
    count = preprocess(
        dicom_dir=dicom_dir,
        output_dir=output_dir,
        db_path=db,
        n_workers=workers,
    )
    
    print(f"\nPreprocessing complete. {count:,} frames ready for labeling.")
    print(f"Run: python scripts/02_label_server.py")