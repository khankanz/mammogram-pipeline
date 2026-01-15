# DICOM Loading & Normalization: A Practical Walkthrough

If you've worked with standard image formats (PNG, JPEG), you're used to pixels being... pixels. Load the file, get an array, done. DICOM is different. Medical imaging carries baggage—decades of it—and understanding *why* requires a quick detour into what these files actually contain.

## The Problem: Raw Pixels Are Meaningless

When a CT scanner captures data, the raw values aren't standardized. Scanner A might represent bone as `2000`, Scanner B as `1847`. Same tissue, different numbers. This is a problem if you're training ML models or comparing scans across institutions.

DICOM solves this with a two-stage normalization pipeline:

1. **Modality LUT** — Convert raw scanner values → standardized units (e.g., Hounsfield Units for CT)
2. **VOI LUT** — Select which range of those units to display (windowing)

Why two stages instead of one monolithic transform? Sanity. You want to read and understand the code. Separating "make values comparable" from "focus on what matters clinically" means you can reason about each independently.

---

## Reading Frames

DICOMs can be single-frame (one image) or multi-frame (a stack, like a video or 3D volume). The array shape tells you which:

```python
def get_frame_count(ds: pydicom.Dataset) -> int:
    """Get number of frames in a DICOM dataset."""
    return int(getattr(ds, 'NumberOfFrames', 1))

def extract_frame(pixel_array: np.ndarray, frame_idx: int = 0) -> np.ndarray:
    """Extract a single frame from pixel array."""
    if pixel_array.ndim == 2: return pixel_array  # Single frame: shape is (rows, cols)
    return pixel_array[frame_idx]  # Multi-frame: shape is (frames, rows, cols)
```

No need to check metadata for frame count when you can infer it from array dimensionality. Single-frame → 2D. Multi-frame → 3D.

---

## Stage 1: Modality LUT (The Universal Translator)

Raw scanner values are gibberish until translated. Two methods exist:

**Method A: Rescale Slope/Intercept**
```
real_value = (raw_pixel × slope) + intercept
```
The slope and intercept ship with the DICOM—they're baked into metadata by the scanner manufacturer.

**Method B: Lookup Table**
Some modalities use a discrete mapping table instead of a linear formula. The DICOM includes this table if needed.

Think of Modality LUT as a universal translator: it converts proprietary scanner gibberish into standardized medical units that mean the same thing regardless of which machine captured the image.

```python
# From normalize_array():
if 'ModalityLUTSequence' in ds: 
    arr = apply_modality_lut(arr, ds)
elif hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
    arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
```

After this step, you have comparable values. A pixel representing lung tissue will have roughly the same value whether it came from a GE scanner in Toronto or a Siemens in Munich.

---

## Stage 2: VOI LUT (The Bandpass Filter)

Now we have standardized units, but the *range* is huge. A CT scan might span -1000 to +3000 Hounsfield Units. Displaying that entire range on an 8-bit image (0-255) crushes all the useful contrast into mush.

VOI LUT is a bandpass filter. You pick:
- **Window Center (WC)**: What value should be medium gray?
- **Window Width (WW)**: How wide a range do we display?

Everything outside that window clips to black or white. Everything inside stretches to fill 0-255.

```python
# Simplified mental model:
def apply_voi_lut_simplified(arr, center, width):
    low = center - width/2
    high = center + width/2
    arr = np.clip(arr, low, high)
    return (arr - low) / (high - low) * 255
```

Where do `WindowCenter` and `WindowWidth` come from? They're stored in DICOM metadata—set by the technician, radiologist, or scanner defaults. Sometimes multiple presets exist (lung window, bone window, soft tissue window). The code grabs the first:

```python
if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
    wc, ww = ds.WindowCenter, ds.WindowWidth
    if isinstance(wc, pydicom.multival.MultiValue): wc = wc[0]
    if isinstance(ww, pydicom.multival.MultiValue): ww = ww[0]
    # ... apply windowing
```

---

## The MONOCHROME1 Quirk

Some old-school radiologists preferred film where bone appeared *dark*. MONOCHROME1 is that legacy. If you encounter it, invert:

```python
if getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2') == 'MONOCHROME1':
    arr = arr.max() - arr  # Flip it to modern conventions
```

Welcome to the 21st century.

---

## Final Safety Net

After all these transforms, your values might be anywhere—0 to 1, -500 to 2000, who knows. One final normalization guarantees 0-255:

```python
arr_min, arr_max = arr.min(), arr.max()
if arr_max - arr_min > 1e-6:
    arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
else: 
    arr = np.zeros_like(arr)  # Edge case: completely flat image
```

This catches DICOMs without window settings, weird edge cases, and any drift from the inversion step.

---

## Converting to PIL

The end goal: a normalized, resized PIL Image ready for downstream ML.

```python
def dicom_to_pil(path: Path, frame_idx: int = 0, size: int = IMAGE_SIZE) -> Optional[Image.Image]:
    """Load DICOM, normalize, return PIL Image or None."""
    try:
        ds = pydicom.dcmread(str(path))
        if not hasattr(ds, 'pixel_array'): return None

        num_frames = getattr(ds, 'NumberOfFrames', 1) 
        if frame_idx >= num_frames: return None

        frame = ds.pixel_array[frame_idx] if num_frames > 1 else ds.pixel_array
        arr = normalize_array(frame, ds)

        return Image.fromarray(arr).resize((size, size), Image.LANCZOS).convert('RGB')
    except Exception as e:
        print(f"Warning: {path}: {e}")
        return None
```

Notes:
- **224×224**: Standard input size for ImageNet-pretrained models
- **LANCZOS**: High-quality resampling. Slower than nearest-neighbor, but better
- **RGB conversion**: Pretrained models expect 3 channels. Grayscale `128` becomes `(128, 128, 128)`

---

## Helper Utilities

**Iterating frames without loading pixels:**

```python
def iter_dicom_frames(path: Path) -> Generator[tuple[int, int], None, None]:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        num_frames = get_frame_count(ds)
        for i in range(num_frames): yield i, num_frames
    except Exception as e:
        logger.warning(f"Cannot read frame info from {path}: {e}")
```

The key here is `stop_before_pixels=True`. DICOM files can be massive—loading pixel data just to check frame count is wasteful. This flag tells pydicom "read metadata only, stop before you hit the heavy stuff." You get frame count, dimensions, modality info—everything except the actual image data.

**Quick metadata lookup:**

```python
def get_dicom_info(path: Path) -> Optional[dict]:
    """Get basic DICOM metadata without loading pixel data."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        attrs = ['Rows', 'Columns', 'PhotometricInterpretation', 
            'Modality', 'StudyDescription', 'SeriesDescription']
        info = {a: getattr(ds, a, None) for a in attrs}
        info['num_frames'] = get_frame_count(ds)
        return AttrDict(info)
    except Exception:
        return None
```

Same optimization pattern. `AttrDict` (from fastcore) lets you access dict keys as attributes—`info.Modality` instead of `info['Modality']`. Minor ergonomic win.

**Thumbnail creation:**

```python
def create_thumbnail(dicom_path: Path, output_path: Path,
                     frame_idx: int = 0, size: int = IMAGE_SIZE) -> Optional[Path]:
    """Create a thumbnail, return output_path or None."""
    img = dicom_to_pil(dicom_path, frame_idx, size)
    if img is None: return None
    mkdir(output_path.parent, exist_ok=True, parents=True)
    img.save(output_path, 'PNG')
    return output_path
```

Thin wrapper around `dicom_to_pil`. Creates parent directories if needed (`mkdir` from fastcore), saves as PNG, returns the path on success or `None` on failure. The `None` return pattern lets you chain with `.filter()` downstream to drop failures cleanly.

---

## Finding DICOMs: Magic Numbers

File extensions lie. A `.dcm` file might be corrupted. A file with no extension might be a valid DICOM. The ground truth is the **magic number**—bytes 128-131 should read `DICM`:

```python
def is_dicom_magic(path: Path) -> bool:
    """Check if DICOM magic number at byte 128."""
    try: 
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False
```

DICOM structure: bytes 0-127 are preamble (often zeros or legacy junk), bytes 128-131 are the magic signature, everything after is metadata and pixel data.

The discovery function combines extension checking with magic number validation:

```python
def find_dicoms(root_dir: Path) -> L:
    """Find all DICOMs, return L of (path, study_id) tuples."""
    root = Path(root_dir)
    dcm = L(globtastic(root, file_glob="*.dcm", func=Path))
    other = (L(globtastic(root, func=Path))
            .filter(Self.is_file())
            .filter(has_dcm_ext, negate=True)
            .filter(is_dicom_magic))
    return (dcm + other).map(with_study_id)
```

`L` is fastcore's monkeypatched list—same list you know, but with chainable methods like `.filter()` and `.map()`. `Self.is_file()` is fastcore shorthand for `lambda x: x.is_file()`.

---

## Parallel Preprocessing

Batch conversion with fastcore's `parallel`:

```python
def preprocess_dicoms(dicom_dir, thumb_dir):
    """Convert all DICOMs to thumbnails in parallel."""
    dicoms = find_dicoms(Path(dicom_dir))
    
    def _process(item):
        dcm, study = item
        out = thumb_dir / f"{study}_{dcm.stem}.png"
        return create_thumbnail(dcm, out)
    
    results = parallel(_process, dicoms, progress=True)
    return L(results).filter()  # Drop None results
```

Nothing fancy—just parallelized I/O with a progress bar.

---

That's the pipeline. Raw scanner values in, normalized thumbnails out, with all the historical medical imaging quirks handled along the way.
