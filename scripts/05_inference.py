#!/usr/bin/env python3
"""Batch classify DICOMs and output study-level CSV.

Also saves predictions to database for review interface.
Includes detailed timing information.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import (
    MODEL_DIR, DICOM_DIR, TEST_DICOM_DIR, THUMBNAIL_DIR, DB_PATH,
    INFERENCE_BATCH_SIZE, IMAGE_SIZE, ensure_dirs
)
from lib.dicom_utils import find_dicoms, dicom_to_pil, get_frame_count
from lib.db import get_db, insert_image


def load_onnx_session(model_path: Path):
    """Load ONNX model."""
    import onnxruntime as ort

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]
    print(f"Using providers: {providers}")

    session = ort.InferenceSession(str(model_path), providers=providers)
    return session


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess PIL image for inference."""
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    return arr


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def batch_predict(session, images: list[np.ndarray]) -> np.ndarray:
    """Run batch prediction."""
    if not images:
        return np.array([])

    batch = np.stack(images).astype(np.float32)
    logits = session.run(None, {'image': batch})[0]
    probs = sigmoid(logits)
    return probs


def run_inference(
    dicom_dir: Path,
    model_path: Path,
    db_path: Path,
    batch_size: int = INFERENCE_BATCH_SIZE,
    save_thumbnails: bool = True
) -> dict:
    """Run inference on DICOM files.

    Returns dict mapping study_id to list of (prob_biopsy, prob_mag) tuples.
    Also saves predictions to database.
    """
    import pydicom

    ensure_dirs()
    session = load_onnx_session(model_path)
    db = get_db(db_path)

    # Timing
    timings = {
        "dicom_discovery": 0,
        "dicom_loading": 0,
        "preprocessing": 0,
        "inference": 0,
        "db_operations": 0,
    }

    # Discover DICOMs
    t0 = time.time()
    dicom_files = list(find_dicoms(dicom_dir))
    timings["dicom_discovery"] = time.time() - t0
    print(f"Found {len(dicom_files)} DICOM files ({timings['dicom_discovery']:.2f}s)")

    if not dicom_files:
        return {}, timings

    results = defaultdict(list)
    now = datetime.now().isoformat()

    batch_inputs = []
    batch_metadata = []  # (study_id, filename, frame_idx, thumb_path)

    total_frames = 0

    for dcm_path, study_id in tqdm(dicom_files, desc="Processing DICOMs"):
        try:
            t0 = time.time()
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            num_frames = get_frame_count(ds)
            timings["dicom_loading"] += time.time() - t0

            for frame_idx in range(num_frames):
                t0 = time.time()
                img = dicom_to_pil(dcm_path, frame_idx, IMAGE_SIZE)
                timings["dicom_loading"] += time.time() - t0

                if img is None:
                    continue

                total_frames += 1

                # Create filename for this frame
                filename = f"{study_id}/{dcm_path.stem}"
                if num_frames > 1:
                    filename = f"{filename}_f{frame_idx}"

                # Save thumbnail if requested
                thumb_path = None
                if save_thumbnails:
                    t0 = time.time()
                    thumb_dir = THUMBNAIL_DIR / study_id
                    thumb_dir.mkdir(parents=True, exist_ok=True)
                    thumb_name = f"{dcm_path.stem}_f{frame_idx}.png" if num_frames > 1 else f"{dcm_path.stem}.png"
                    thumb_path = thumb_dir / thumb_name
                    img.save(thumb_path)
                    timings["preprocessing"] += time.time() - t0

                t0 = time.time()
                arr = preprocess_image(img)
                timings["preprocessing"] += time.time() - t0

                batch_inputs.append(arr)
                batch_metadata.append((study_id, filename, frame_idx, str(thumb_path) if thumb_path else ""))

                if len(batch_inputs) >= batch_size:
                    t0 = time.time()
                    probs = batch_predict(session, batch_inputs)
                    timings["inference"] += time.time() - t0

                    t0 = time.time()
                    for i, (sid, fname, fidx, tpath) in enumerate(batch_metadata):
                        results[sid].append((float(probs[i, 0]), float(probs[i, 1])))

                        # Save to database
                        if tpath:
                            insert_image(db, fname, sid, tpath, fidx)
                            db["labels"].update(
                                db.execute("SELECT id FROM labels WHERE filename = ?", [fname]).fetchone()[0],
                                {
                                    "confidence_biopsy": float(probs[i, 0]),
                                    "confidence_mag": float(probs[i, 1]),
                                    "predicted_at": now,
                                }
                            )
                    timings["db_operations"] += time.time() - t0

                    batch_inputs = []
                    batch_metadata = []

        except Exception as e:
            print(f"Warning: Failed to process {dcm_path}: {e}")

    # Process remaining batch
    if batch_inputs:
        t0 = time.time()
        probs = batch_predict(session, batch_inputs)
        timings["inference"] += time.time() - t0

        t0 = time.time()
        for i, (sid, fname, fidx, tpath) in enumerate(batch_metadata):
            results[sid].append((float(probs[i, 0]), float(probs[i, 1])))

            if tpath:
                insert_image(db, fname, sid, tpath, fidx)
                db["labels"].update(
                    db.execute("SELECT id FROM labels WHERE filename = ?", [fname]).fetchone()[0],
                    {
                        "confidence_biopsy": float(probs[i, 0]),
                        "confidence_mag": float(probs[i, 1]),
                        "predicted_at": now,
                    }
                )
        timings["db_operations"] += time.time() - t0

    timings["total_frames"] = total_frames
    return dict(results), timings


def aggregate_to_study_level(results: dict, threshold: float = 0.5) -> list[dict]:
    """Aggregate image-level predictions to study level."""
    study_results = []

    for study_id, predictions in results.items():
        if not predictions:
            continue

        probs_biopsy = [p[0] for p in predictions]
        probs_mag = [p[1] for p in predictions]

        max_biopsy = max(probs_biopsy)
        max_mag = max(probs_mag)

        study_results.append({
            'study_id': study_id,
            'has_biopsy_tool': 1 if max_biopsy >= threshold else 0,
            'has_mag_view': 1 if max_mag >= threshold else 0,
            'max_prob_biopsy': round(max_biopsy, 4),
            'max_prob_mag': round(max_mag, 4),
            'image_count': len(predictions),
        })

    study_results.sort(key=lambda x: x['study_id'])
    return study_results


def write_csv(results: list[dict], output_path: Path):
    """Write results to CSV."""
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


def main():
    parser = argparse.ArgumentParser(description="Batch classify DICOMs")
    parser.add_argument("--input-dir", type=Path, help="Directory containing DICOM files")
    parser.add_argument("--output", type=Path, default=Path("results.csv"), help="Output CSV path")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "model.onnx", help="ONNX model path")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path")
    parser.add_argument("--batch-size", type=int, default=INFERENCE_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test", action="store_true", help=f"Use test directory ({TEST_DICOM_DIR})")
    parser.add_argument("--no-thumbnails", action="store_true", help="Skip saving thumbnails")

    args = parser.parse_args()

    if args.test:
        input_dir = TEST_DICOM_DIR
    elif args.input_dir:
        input_dir = args.input_dir
    else:
        input_dir = DICOM_DIR

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not args.model.exists():
        print(f"Error: Model not found: {args.model}")
        print("Run export first: python scripts/04_export.py")
        sys.exit(1)

    print(f"Input: {input_dir}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Database: {args.db}")
    print(f"Threshold: {args.threshold}")
    print()

    start_time = time.time()

    results, timings = run_inference(
        input_dir, args.model, args.db, args.batch_size,
        save_thumbnails=not args.no_thumbnails
    )

    study_results = aggregate_to_study_level(results, args.threshold)
    write_csv(study_results, args.output)

    # Print timing summary
    total_time = time.time() - start_time
    total_frames = timings.get("total_frames", 0)

    print(f"\n{'='*50}")
    print(f"TIMING SUMMARY")
    print(f"{'='*50}")
    print(f"Total time:        {total_time:.2f}s")
    print(f"Total frames:      {total_frames}")
    if total_frames > 0:
        print(f"Throughput:        {total_frames / total_time:.1f} frames/sec")
    print(f"\nBreakdown:")
    print(f"  DICOM discovery: {timings['dicom_discovery']:.2f}s")
    print(f"  DICOM loading:   {timings['dicom_loading']:.2f}s")
    print(f"  Preprocessing:   {timings['preprocessing']:.2f}s")
    print(f"  Inference:       {timings['inference']:.2f}s")
    print(f"  DB operations:   {timings['db_operations']:.2f}s")
    print(f"{'='*50}")

    # Print results summary
    biopsy_count = sum(1 for r in study_results if r['has_biopsy_tool'])
    mag_count = sum(1 for r in study_results if r['has_mag_view'])
    both_count = sum(1 for r in study_results if r['has_biopsy_tool'] and r['has_mag_view'])

    print(f"\nRESULTS SUMMARY")
    print(f"Studies processed: {len(study_results)}")
    print(f"Studies with Biopsy Tool: {biopsy_count}")
    print(f"Studies with Mag View: {mag_count}")
    print(f"Studies with Both: {both_count}")

    print(f"\nNext: Review predictions with: python scripts/07_review.py")


if __name__ == "__main__":
    main()
