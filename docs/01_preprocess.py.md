# Preprocessing Pipeline: DICOMs to Training-Ready Thumbnails

This script is the entry point—raw DICOMs go in, normalized thumbnails and database records come out. It's designed to be run once at project start, then incrementally as new data arrives.

## The Timer Utility

Simple throughput tracking. Nothing fancy, but useful for understanding where time goes:

```python
class Timer:
    "Simple timer that tracks elapsed time and throughput."
    def __init__(self, name=''):
        self.name, self.start, self.n = name, time.perf_counter(), 0
    
    def __repr__(self):
        elapsed = time.perf_counter() - self.start
        rate = self.n / elapsed if elapsed > 0 else 0
        return f"{self.name}: {self.n:,} items in {elapsed:.1f}s ({rate:.1f}/sec)"
    
    def add(self, n=1): self.n += n; return self
```

Usage: `t = Timer("Processing"); ...; t.add(100); print(t)` → `Processing: 100 items in 2.3s (43.5/sec)`

---

## Frame Naming Strategy

Multi-frame DICOMs need unique identifiers per frame. Single-frame DICOMs don't need the suffix:

```python
def _frame_filename(study_id, stem, frame_idx, num_frames):
    "Generate unique filenames for a frame."
    base = f"{study_id}/{stem}"
    return f"{base}_frame{frame_idx:04d}" if num_frames > 1 else base
```

`study_001/image_a` for single-frame, `study_001/image_b_frame0003` for multi-frame. The zero-padding (`{frame_idx:04d}`) ensures proper sorting up to 9999 frames.

---

## Processing a Single Frame

Each frame gets its own thumbnail. The function is idempotent—if the thumbnail exists, skip the work:

```python
def _process_frame(dcm_path, study_id, frame_idx, num_frames, out_dir):
    "Process a single frame, return dict or None"
    filename = _frame_filename(study_id, dcm_path.stem, frame_idx, num_frames)
    thumb_name = filename.replace("/", "_") + ".png"
    thumb_path = out_dir / thumb_name

    # Idempotent: skip if exists
    if thumb_path.exists(): 
        return dict(filename=filename, study_id=study_id, thumbnail_path=str(thumb_path))
    
    # Create thumbnail
    if create_thumbnail(dcm_path, thumb_path, frame_idx, IMAGE_SIZE): 
        return dict(filename=filename, study_id=study_id, 
                    thumbnail_path=str(thumb_path), frame_number=frame_idx)
    return None
```

Returns a dict on success (for database insertion), `None` on failure. The `None` pattern lets us filter failures downstream with `.filter()`.

---

## Processing a DICOM File

A single DICOM might contain multiple frames. We need the pixel data here—can't use `stop_before_pixels=True`:

```python
def process_dicom(dcm_path, study_id, out_dir):
    "Process a single DICOM file, return list of frame dicts"
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=False)
        num_frames = get_frame_count(ds)
    except Exception as e:
        logger.warning(f"Cannot read {dcm_path}: {e}")
        return []
    
    results = []
    for i in range(num_frames):
        r = _process_frame(dcm_path, study_id, i, num_frames, out_dir)
        if r: results.append(r)
    return results
```

Returns a list of frame dicts. Empty list on failure—no exceptions bubble up, just logged warnings.

---

## Parallel Processing Wrapper

`fastcore.parallel` needs a single-argument function. We use `partial` to bind the output directory, and suppress warnings inside worker processes:

```python
def _process_wrapper(item, out_dir):
    "Wrapper for parallel - unpacks tuple"
    warnings.simplefilter("ignore")
    dcm_path, study_id = item
    return process_dicom(dcm_path, study_id, out_dir)
```

The `warnings.simplefilter("ignore")` prevents pydicom's deprecation warnings from flooding your terminal during parallel execution.

---

## The Main Pipeline

Three phases: discover, process, insert.

```python
def preprocess(dicom_dir, output_dir, db_path, n_workers=NUM_WORKERS):
    "Process all DICOMs in directory tree"
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Discovery
    t_discover = Timer("Discover")
    dicoms = find_dicoms(dicom_dir)
    t_discover.add(len(dicoms))
    
    if not dicoms: print("No DICOMs found. Exiting."); return 0

    # Phase 2: Parallel processing
    t_process = Timer("Processing")
    results = parallel(
        partial(_process_wrapper, out_dir=output_dir),
        dicoms, 
        n_workers=n_workers, 
        progress=True, 
        threadpool=False  # Processes, not threads—CPU-bound work
    )
    
    # Flatten: list of lists → single list
    all_frames = L(results).concat()
    t_process.add(len(all_frames))

    # Phase 3: Database insertion
    t_db = Timer("DB Insert")
    db = get_db(db_path)

    inserted = 0
    for frame in progress_bar(all_frames, leave=False):
        if not image_exists(db, frame["filename"]):
            insert_image(db, frame["filename"], frame["study_id"], 
                frame["thumbnail_path"], frame["frame_number"])
            inserted += 1
    t_db.add(inserted)
    
    return len(all_frames)
```

Key details:

- **`threadpool=False`**: DICOM processing is CPU-bound (pixel normalization, image resizing). Processes sidestep Python's GIL. Threads would serialize the work.
- **`L(results).concat()`**: Each DICOM returns a list of frames. `concat()` flattens `[[frame1, frame2], [frame3], ...]` into `[frame1, frame2, frame3, ...]`.
- **Idempotent insertion**: `image_exists()` check prevents duplicates if you re-run the script.

---

## The CLI: Zero Boilerplate

`@call_parse` from fastcore introspects your function signature and builds a CLI automatically:

```python
@call_parse
def main(
    dicom_dir:Path=DICOM_DIR,      # Directory containing DICOM files
    output_dir:Path=THUMBNAIL_DIR, # Output directory for thumbnails
    db:Path=DB_PATH,               # SQLite database path
    workers:int=NUM_WORKERS,       # Number of parallel workers
    test:bool_arg=False            # Use test directory instead
):
    "Preprocess DICOM files to thumbnails"
    # ...
```

What `@call_parse` does:

| Function Signature | CLI Behavior |
|---|---|
| Parameter names | Become `--arg-name` flags |
| Type annotations | Determine argument types |
| Default values | Make args optional |
| Inline comments | Become `--help` text |

This eliminates ~20 lines of argparse boilerplate.

---

## Running It

```bash
python scripts/01_preprocess.py --dicom_dir samp_ds
```

Real output from a 50-file test dataset:

```
Scanning samp_ds for DICOMs files...

Processing with 8 workers...
 |-----------------------------------------------------| 0.00% [0/50 00:00<?]
Warning: samp_ds/JPEG2000-embedded-sequence-delimiter.dcm: Unable to decode as exceptions were raised by all available plugins:
  pillow: Image size (3811783737344 pixels) exceeds limit of 178956970 pixels, could be decompression bomb DOS attack.
Warning: samp_ds/MR_truncated.dcm: The number of bytes of pixel data is less than expected (8130 vs 8192 bytes) - the dataset may be corrupted...
Warning: samp_ds/GDCMJ2K_TextGBR.dcm: Unable to decode as exceptions were raised by all available plugins:
  pillow: broken data stream when reading image file
 |███--------------------------------------------------| 6.00% [3/50 00:00<00:01]
Cannot read samp_ds/badVR.dcm: invalid literal for int() with base 10: '1A'
Cannot read samp_ds/no_meta.dcm: File is missing DICOM File Meta Information header or the 'DICM' prefix is missing...
Cannot read samp_ds/rtstruct.dcm: File is missing DICOM File Meta Information header or the 'DICM' prefix is missing...
    Processing: 95 items in 2.6s (36.1/sec)                                         

Updating database....
  DB Insert: 95 items in 0.4s (213.4/sec)                                           
  Total in database: 95

==================================================
SUMMARY
==================================================
  Discover: 50 items in 3.1s (16.2/sec)
  Processing: 95 items in 3.1s (30.9/sec)
  DB Insert: 95 items in 0.4s (213.2/sec)
==================================================

Preprocessing complete. 95 frames ready for labeling.
Run: python scripts/02_label_server.py
```

Notice what happens with bad files:

- **Truncated data**: Warns but continues (`MR_truncated.dcm`)
- **Corrupted compression**: Warns but continues (`GDCMJ2K_TextGBR.dcm`)
- **Decompression bomb**: Caught by pillow's safety check, skipped
- **Missing headers**: Logged and skipped (`no_meta.dcm`, `rtstruct.dcm`)
- **Invalid metadata**: Logged and skipped (`badVR.dcm`)

50 DICOMs discovered → 95 frames extracted (some multi-frame DICOMs) → failures logged, survivors inserted. The pipeline is defensive—bad files don't crash the batch.

---

That's the preprocessing pipeline. DICOMs discovered, thumbnails generated in parallel, database populated with hold-out splits already assigned. Ready for labeling.
