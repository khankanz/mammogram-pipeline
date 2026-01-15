"""DICOM loading, normalization, and thumbnail creation with proper VOI LUT support."""

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
from PIL import Image
from pathlib import Path
from typing import Generator, Optional
from .config import IMAGE_SIZE
from fastcore.all import mkdir


def get_frame_count(ds: pydicom.Dataset) -> int:
    """Get number of frames in a DICOM dataset."""
    return int(getattr(ds, 'NumberOfFrames', 1)) # What are we grabbing frame count?


def extract_frame(pixel_array: np.ndarray, frame_idx: int = 0) -> np.ndarray:
    """Extract a single frame from pixel array."""
    if pixel_array.ndim == 2: return pixel_array
    return pixel_array[frame_idx]
    # We can infer num_frames from array shape, single-frame DICOMS shape is rows, col i.e. 2D, multiframe DICOM frames, rows, cols i.e. 3D


def normalize_array(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Apply DICOM-standard normalization: Modality LUT -> VOI LUT -> photometric handling."""
    # Keep original dtype for LUT functions, convert to float after
    # Apply Modality LUT (RescaleSlope/Intercept)
    # This converts stored values to output units (e.g., Hounsfield units)
    arr = arr.astype(np.float64)
    if 'ModalityLUTSequence' in ds: arr = apply_modality_lut(arr, ds)     # Okay so I looked into this: it applies a modality lookup table or rescale operations to arr, how does it know what to do? it's fed a dataset containing a Modality LUT Module? Where is this coming from?
    elif hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    #Okay so, inp is arr? we have done what? convert raw scanner values to comparable values. two methods to do this: rescale/intercept -> real_value = (raw_pixel * slope) + intercept BUT where is the slope and intercept #s coming from?
    # Method 2: Lookup Table. I'm assuming dicom image comes with it's own lookup table based off of scanner? I believe so
    # Great analogy is: a universal translator that converts raw scanner gibberish into standardized medical units that mean the same thing across all machines
    # Apply VOI LUT (windowing)
    # This maps the output range to display values
    if 'VOILUTSequence' in ds: arr = apply_voi_lut(arr, ds)
    # this function is choosing what to focus on or value of interest
        # how is this done? window center (WC) = what value should be medium grey?
        # window width (WW) = how wide a range do we display?
        # We're building a bandpass filter around the VOI; for instance say lung is -600 HU, we decide this is our medium grey. We pick a region around it, we stretch that rhange to 0-255, everything else Clip to black (0) or white (255)
        # BUT how is lung chosen? IT's stored in the DICOM metadata, ds.WindowCenter, ds.WindowWidth. The technician and radiologist see these during capture, sometimes there a multiple presets baked in
        # you can literally write it out as: def apply_voi_lut_simplified(arr, center, width): low = center - windth/2; high = center + width/2; arr = np.clip(arr, low, high); return (arr-low)/(high-low)*255
        # Keep in mind; for linear window case only; edgecases include WindowCenter is a list, non-linear LUT, no WindowCenter/Width exists, different VOI LUT functions i.e. linear, sigmoid etc.
    elif hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
        wc, ww = ds.WindowCenter, ds.WindowWidth
        # Handle multi-valued window settings (use first)
        if isinstance(wc, pydicom.multival.MultiValue): wc = wc[0]
        if isinstance(ww, pydicom.multival.MultiValue): ww = ww[0]
        wc, ww = float(wc), float(ww)
        low, high = wc - ww/2, wc + ww/2
        arr = np.clip(arr, low, high)
    # Handle photometric interpretation (MONOCHROME1 = inverted)
    # Historical reasons; some radiologist prefer film where bone was dark. SO if scans are old-school, we say yo bro 21st is calling. tho tbh, I'm used to this from movies
    if getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2') == 'MONOCHROME1':
        arr = arr.max() - arr
    # Normalize to 0-255
    # This is a safety net, after all these LUT transformations,we might end up with
        # 0-1??! rescale to 0-255
        # -500 to +2000? Rescale to 0-255
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min > 1e-6:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    else: 
        arr = np.zeros_like(arr)
    # this last catch was to account for: DICOMS without WindowCenter,Width, weird values that don't map, MONOCHROME1 inversion that might shift range

    return arr.astype(np.uint8)


def dicom_to_pil(path: Path, frame_idx: int = 0, size: int = IMAGE_SIZE) -> Optional[Image.Image]:
    """Load DICOM, normalize, return PIL Image or None."""
    try:
        ds = pydicom.dcmread(str(path)) # ds now contains the entire DICOM dataset; pixel data + metadata
        if not hasattr(ds, 'pixel_array'): return None

        num_frames = getattr(ds, 'NumberOfFrames', 1) 
        if frame_idx >= num_frames: return None # IF we asked for frame #50 and only 10 frames exists, return None

        frame = ds.pixel_array[frame_idx] if num_frames > 1 else ds.pixel_array
        arr = normalize_array(frame, ds) # let's work all the magic here; converting raw sensor values, adjusting brightness and constrast, inverse corrections

        return Image.fromarray(arr).resize((size,size), Image.LANCZOS).convert('RGB')
        # size = 224x224; Image.LANCZOS a high-quality resizing algorithm. slower but better than nearest neighbors or bilinear
        # imagenet pretrained models expect RGB, we effectively duplicate values into three channels i.e. Grascale 128 becomes RGB(128,128,128)
    except Exception as e:
        print(f"Warning: {path}: {e}")
        return None

import logging
logger = logging.getLogger(__name__)

def iter_dicom_frames(path: Path) -> Generator[tuple[int, int], None, None]:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        num_frames = get_frame_count(ds)
        for i in range(num_frames): yield i, num_frames
    except Exception as e:
        logger.warning(f"Cannot read frame info from {path}: {e}")

from fastcore.basics import AttrDict
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


def create_thumbnail(dicom_path: Path, output_path: Path,
                     frame_idx: int = 0, size: int = IMAGE_SIZE) -> Optional[Path]:
    """Create a thumbnail, return output_path or None."""
    img = dicom_to_pil(dicom_path, frame_idx, size)
    if img is None: return None
    mkdir(output_path.parent, exist_ok=True, parents=True)
    img.save(output_path, 'PNG')
    return output_path
    
from fastcore.foundation import L
from fastcore.xtras import globtastic
from fastcore.basics import Self
from pathlib import Path

# Most file foramts embed a hidden "signature" in the first few bytes of the file.
# We can read this to find out what type of file it is irrespective of the file extension.
# Bytes 0-127; preamble zeros or legacy junk, Bytes 128-131: D I C M (magic number), Bytes 132+ [actual DICOM metadata + pixel data]
def is_dicom_magic(path: Path) -> bool:
    """Check if DICOM magic number at byte 128"""
    try: 
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False

def has_dcm_ext(p: Path) -> bool: return p.suffix.lower() == '.dcm'
def with_study_id(p: Path) -> tuple[Path, str]: return (p, p.parent.name)

def find_dicoms(root_dir: Path) -> L:
    """Find all DICOMs, return L of (path, study_id) tuples."""
    root = Path(root_dir)
    dcm = L(globtastic(root, file_glob="*.dcm", func=Path))
    other = (L(globtastic(root, func=Path))
            .filter(Self.is_file())
            .filter(has_dcm_ext, negate=True)
            .filter(is_dicom_magic))
    return (dcm + other).map(with_study_id)

import os
from fastcore.parallel import parallel

def preprocess_dicoms(dicom_dir, thumb_dir):
    """Convert all DICOMS to thumbnails in parallel."""
    dicoms = find_dicoms(Path(dicom_dir))
    
    def _process(item):
        dcm, study = item
        out = thumb_dir / f"{study}_{dcm.stem}.png"
        return create_thumbnail(dcm, out)
    
    results = parallel(_process, dicoms, progress=True)
    return L(results).filter()