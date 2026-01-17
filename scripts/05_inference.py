#!/usr/bin/env python3
"""Production batch inference: classify DICOMs → SQLite database + CSV export.

This script is PRODUCTION ONLY. It does NOT touch the development database.
It operates on new DICOMs you've never seen before.

Output:
- PRIMARY: SQLite database (results.db) - image-level predictions, resumable
- DERIVED: CSV file (results.csv) - study-level aggregation for humans/Excel

Design principles:
1. Standalone - no dependency on development database or training workflow
2. Resumable - SQLite tracks progress, can restart from interruption  
3. Fast - parallel DICOM loading, batched ONNX inference
4. Study-level output - radiologists think in studies, not images

Usage:
    # Basic: classify all DICOMs in a directory
    python scripts/05_inference.py --input /path/to/dicoms
    
    # Resume interrupted job
    python scripts/05_inference.py --input /path/to/dicoms --resume
    
    # Custom output paths
    python scripts/05_inference.py --input /path/to/dicoms --db results.db --output results.csv
    
    # Test mode with sample data
    python scripts/05_inference.py --test
"""

import sys, time, logging
from pathlib import Path
from datetime import datetime
from functools import partial
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from sqlite_utils import Database

from fastcore.all import *
from fastcore.script import call_parse, bool_arg
from fastprogress import progress_bar

from lib.config import MODEL_DIR, DICOM_DIR, TEST_DICOM_DIR, IMAGE_SIZE, ensure_dirs
from lib.dicom_utils import find_dicoms, dicom_to_pil, get_frame_count

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

BATCH_SIZE = 32
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Results database schema - separate from development database!
# This is for tracking inference progress and storing predictions on production data.
# Pattern borrowed from lib/db.py - schema + indexes defined together, applied in _init_db
_results_schema = dict(
    id=int, filename=str, study_id=str, frame_number=int,
    prob_biopsy=float, prob_mag=float, predicted_at=str
)

_results_indexes = [
    (["filename"], {"unique": True}),   # Fast lookup for resume, prevent duplicates
    (["study_id"], {}),                  # Fast aggregation to study level
    (["prob_biopsy"], {}),               # Query "show me high-confidence biopsy"
    (["prob_mag"], {}),                  # Query "show me high-confidence mag"
    (["predicted_at"], {}),              # Query by time range
]

# =============================================================================
# ONNX SESSION
# =============================================================================

def load_session(model_path):
    "Load ONNX model with best available provider (CUDA > CPU)"
    import onnxruntime as ort
    
    # Try CUDA first, fall back to CPU
    # This is the pattern: list your preferences, filter to what's available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]
    
    print(f"ONNX providers: {providers}")
    return ort.InferenceSession(str(model_path), providers=providers)

# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess(img):
    "PIL Image → normalized numpy array ready for ONNX. Shape: (3, 224, 224)"
    # Step 1: Convert to float 0-1 range
    # Neural networks train better with small centered values.
    # Raw pixels 0-255 would blow up gradients in early layers.
    arr = np.array(img).astype(np.float32) / 255.0
    
    # Step 2: ImageNet normalization
    # The pretrained model learned features on ImageNet, which has these specific
    # per-channel statistics. We normalize our data to match that distribution.
    # Without this, the model sees "out of distribution" inputs and performs worse.
    # It's like speaking the model's native language vs broken phrases.
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    
    # Step 3: HWC → CHW
    # PIL/numpy store images as (Height, Width, Channels) - rows of RGB pixels
    # PyTorch/ONNX expect (Channels, Height, Width) - separate R, G, B planes
    # Same data, different memory layout
    return arr.transpose(2, 0, 1)


def sigmoid(x):
    "Logits → probabilities. Numerically stable version."
    # Sigmoid squashes any real number to (0, 1)
    # sigmoid(219) ≈ 1.0, sigmoid(-268) ≈ 0.0
    # 
    # The clip prevents overflow: exp(710) overflows float64.
    # In practice, sigmoid(|x| > 20) is already 0 or 1 for all purposes.
    # The clip is belt-and-suspenders defensive programming.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# =============================================================================
# BATCH INFERENCE
# =============================================================================

def predict_batch(session, images):
    "Run ONNX inference on a batch of preprocessed images. Returns (N, 2) probabilities."
    if not images: return np.array([])
    
    # Stack list of (3, 224, 224) → single (N, 3, 224, 224) batch
    batch = np.stack(images).astype(np.float32)
    
    # session.run(output_names, input_dict)
    # None = return all outputs
    # 'image' = the input name we defined during ONNX export
    logits = session.run(None, {'image': batch})[0]
    
    return sigmoid(logits)

# =============================================================================
# RESULTS DATABASE
# =============================================================================
# Why SQLite instead of JSONL for 500K images?
# 1. Resume: WHERE predicted_at IS NULL instantly finds unprocessed images
# 2. Query: "Show all biopsy=YES" is one SQL statement vs parsing entire file
# 3. Concurrent: WAL mode handles writes safely; JSONL risks corruption
# 4. Speed: ~295K inserts/sec batched; comparable to JSONL append
# 
# Simon Willison: "If you need local structured data storage, SQLite should be your default."

def _init_results_db(db):
    "Create table and indexes if needed. Pattern from lib/db.py"
    if "predictions" in db.table_names(): return
    db["predictions"].create(_results_schema, pk="id")
    for cols, kw in _results_indexes: db["predictions"].create_index(cols, **kw)


def get_results_db(path):
    "Get or create results database. Separate from development database!"
    db = Database(str(path))
    db.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    _init_results_db(db)
    return db


def is_processed(db, filename):
    "Check if we already predicted this file (for resume capability)"
    return db.execute(
        "SELECT 1 FROM predictions WHERE filename = ? LIMIT 1", [filename]
    ).fetchone() is not None


def save_predictions(db, preds):
    "Bulk insert predictions. preds is list of dicts."
    if not preds: return
    db["predictions"].insert_all(preds, ignore=True)

# =============================================================================
# STUDY-LEVEL AGGREGATION
# =============================================================================

def aggregate_studies(db, threshold=0.5):
    """Aggregate image-level predictions to study-level results.
    
    Logic: If ANY image in a study has prob >= threshold, flag the study.
    We use MAX because biopsy markers might only appear in 2 of 15 images.
    Averaging would dilute the signal.
    
    Returns list of dicts ready for CSV.
    """
    rows = list(db["predictions"].rows)
    
    # Group by study_id
    studies = defaultdict(list)
    for r in rows:
        studies[r["study_id"]].append((r["prob_biopsy"], r["prob_mag"]))
    
    results = []
    for study_id, preds in studies.items():
        max_biopsy = max(p[0] for p in preds)
        max_mag = max(p[1] for p in preds)
        
        results.append({
            'study_id': study_id,
            'has_biopsy_tool': 1 if max_biopsy >= threshold else 0,
            'has_mag_view': 1 if max_mag >= threshold else 0,
            'max_prob_biopsy': round(max_biopsy, 4),
            'max_prob_mag': round(max_mag, 4),
            'image_count': len(preds),
        })
    
    return sorted(results, key=lambda x: x['study_id'])


def write_csv(results, output_path):
    "Write study-level results to CSV"
    import csv
    
    if not results:
        print("No results to write")
        return
    
    fieldnames = ['study_id', 'has_biopsy_tool', 'has_mag_view',
                  'max_prob_biopsy', 'max_prob_mag', 'image_count']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Wrote {len(results)} studies to {output_path}")

# =============================================================================
# SINGLE DICOM PROCESSING
# =============================================================================

def process_dicom(item, session):
    """Process one DICOM file, return list of (filename, study_id, frame_idx, probs).
    
    Handles multi-frame DICOMs by iterating through all frames.
    Returns empty list on failure (defensive - don't crash the batch).
    """
    import pydicom
    dcm_path, study_id = item
    results = []
    
    try:
        # We need pixel data here (no stop_before_pixels)
        ds = pydicom.dcmread(str(dcm_path))
        num_frames = get_frame_count(ds)
        
        for frame_idx in range(num_frames):
            # Generate unique filename per frame
            filename = f"{study_id}/{dcm_path.stem}"
            if num_frames > 1:
                filename = f"{filename}_frame{frame_idx:04d}"
            
            # Load and preprocess
            img = dicom_to_pil(dcm_path, frame_idx, IMAGE_SIZE)
            if img is None:
                continue
            
            arr = preprocess(img)
            results.append((filename, study_id, frame_idx, arr))
            
    except Exception as e:
        logger.warning(f"Failed to process {dcm_path}: {e}")
    
    return results

# =============================================================================
# MAIN INFERENCE LOOP
# =============================================================================

class Timer:
    "Simple timer for throughput tracking"
    def __init__(self, name=''):
        self.name, self.start, self.n = name, time.perf_counter(), 0
    
    def __repr__(self):
        elapsed = time.perf_counter() - self.start
        rate = self.n / elapsed if elapsed > 0 else 0
        return f"{self.name}: {self.n:,} items in {elapsed:.1f}s ({rate:.1f}/sec)"
    
    def add(self, n=1): self.n += n; return self


def run_inference(dicom_dir, model_path, db_path, bs=BATCH_SIZE, resume=False):
    """Main inference pipeline.
    
    Args:
        dicom_dir: Directory containing DICOM files
        model_path: Path to ONNX model
        db_path: Path to results SQLite database
        bs: Batch size for inference
        resume: If True, skip already-processed files
    
    Returns:
        Number of images processed
    """
    ensure_dirs()
    
    # Load model
    print(f"Loading model from {model_path}...")
    session = load_session(model_path)
    
    # Setup results database
    db = get_results_db(db_path)
    
    # Discover DICOMs
    t_discover = Timer("Discovery")
    print(f"Scanning {dicom_dir} for DICOMs...")
    dicoms = find_dicoms(dicom_dir)
    t_discover.add(len(dicoms))
    print(f"  Found {len(dicoms)} DICOM files")
    
    if not dicoms:
        print("No DICOMs found. Exiting.")
        return 0
    
    # Filter already processed (if resuming)
    if resume:
        before = len(dicoms)
        # This is O(n) database lookups, but for resume capability it's worth it
        # Alternative: load all processed filenames into a set first
        processed = {r["filename"] for r in db["predictions"].rows}
        dicoms = L(dicoms).filter(lambda x: f"{x[1]}/{x[0].stem}" not in processed)
        print(f"  Resuming: {before - len(dicoms)} already processed, {len(dicoms)} remaining")
    
    if not dicoms:
        print("All images already processed.")
        return 0
    
    # Process in batches
    t_process = Timer("Processing")
    t_inference = Timer("Inference")
    
    batch_inputs = []   # preprocessed arrays
    batch_meta = []     # (filename, study_id, frame_idx)
    total_processed = 0
    
    print(f"\nProcessing {len(dicoms)} DICOMs...")
    
    for dcm_path, study_id in progress_bar(dicoms):
        # Process this DICOM (may have multiple frames)
        frames = process_dicom((dcm_path, study_id), session)
        
        for filename, sid, frame_idx, arr in frames:
            batch_inputs.append(arr)
            batch_meta.append((filename, sid, frame_idx))
            
            # When batch is full, run inference
            if len(batch_inputs) >= bs:
                t_inference_start = time.perf_counter()
                probs = predict_batch(session, batch_inputs)
                t_inference.add(len(batch_inputs))
                t_inference.n  # just to track
                
                # Save to database
                now = datetime.now().isoformat()
                preds = [
                    {
                        "filename": meta[0],
                        "study_id": meta[1],
                        "frame_number": meta[2],
                        "prob_biopsy": float(probs[i, 0]),
                        "prob_mag": float(probs[i, 1]),
                        "predicted_at": now,
                    }
                    for i, meta in enumerate(batch_meta)
                ]
                save_predictions(db, preds)
                
                total_processed += len(batch_inputs)
                t_process.add(len(batch_inputs))
                
                # Clear batch
                batch_inputs = []
                batch_meta = []
    
    # Process remaining batch
    if batch_inputs:
        probs = predict_batch(session, batch_inputs)
        t_inference.add(len(batch_inputs))
        
        now = datetime.now().isoformat()
        preds = [
            {
                "filename": meta[0],
                "study_id": meta[1],
                "frame_number": meta[2],
                "prob_biopsy": float(probs[i, 0]),
                "prob_mag": float(probs[i, 1]),
                "predicted_at": now,
            }
            for i, meta in enumerate(batch_meta)
        ]
        save_predictions(db, preds)
        
        total_processed += len(batch_inputs)
        t_process.add(len(batch_inputs))
    
    print(f"\n{t_process}")
    print(f"{t_inference}")
    
    return total_processed

# =============================================================================
# CLI
# =============================================================================

@call_parse
def main(
    input:Path=None,                         # Directory containing DICOM files
    db:Path=Path("results.db"),              # Results database path (PRIMARY output)
    output:Path=Path("results.csv"),         # Study-level CSV export (derived from db)
    model:Path=MODEL_DIR/"model.onnx",       # ONNX model path
    batch_size:int=BATCH_SIZE,               # Batch size for inference
    threshold:float=0.75,                     # Classification threshold
    test:bool_arg=False,                     # Use test directory
    resume:bool_arg=False,                   # Resume interrupted job
):
    """Production batch inference: DICOMs → SQLite (primary) + CSV (derived).
    
    This is PRODUCTION inference - completely separate from development workflow.
    Results go to a new database, not your training database.
    
    The SQLite database stores image-level predictions and enables resume.
    The CSV is a study-level aggregation for human review.
    """
    # Determine input directory
    if test:
        input_dir = TEST_DICOM_DIR
    elif input:
        input_dir = input
    else:
        input_dir = DICOM_DIR
    
    # Validate inputs
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not model.exists():
        print(f"Error: Model not found: {model}")
        print("Export model first: python scripts/04_export.py")
        sys.exit(1)
    
    # Print configuration
    print(f"{'='*60}")
    print("PRODUCTION INFERENCE")
    print(f"{'='*60}")
    print(f"Input:     {input_dir}")
    print(f"Model:     {model}")
    print(f"Database:  {db} (primary output)")
    print(f"CSV:       {output} (study-level export)")
    print(f"Threshold: {threshold}")
    print(f"Resume:    {resume}")
    print(f"{'='*60}\n")
    
    # Run inference
    start = time.perf_counter()
    count = run_inference(input_dir, model, db, batch_size, resume)
    elapsed = time.perf_counter() - start
    
    # Aggregate to study level
    print("\nAggregating to study level...")
    results_db = get_results_db(db)
    study_results = aggregate_studies(results_db, threshold)
    
    # Write CSV
    write_csv(study_results, output)
    
    # Summary
    biopsy_count = sum(1 for r in study_results if r['has_biopsy_tool'])
    mag_count = sum(1 for r in study_results if r['has_mag_view'])
    both_count = sum(1 for r in study_results if r['has_biopsy_tool'] and r['has_mag_view'])
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Images processed:        {count:,}")
    print(f"Total time:              {elapsed:.1f}s")
    if count > 0:
        print(f"Throughput:              {count/elapsed:.1f} images/sec")
    print(f"Studies classified:      {len(study_results)}")
    print(f"  With biopsy tool:      {biopsy_count}")
    print(f"  With mag view:         {mag_count}")
    print(f"  With both:             {both_count}")
    print(f"{'='*60}")