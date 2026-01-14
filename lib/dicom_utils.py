"""DICOM loading, normalization, and thumbnail creation with proper VOI LUT support."""

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
from PIL import Image
from pathlib import Path
from typing import Generator, Optional
from .config import IMAGE_SIZE


def get_frame_count(ds: pydicom.Dataset) -> int:
    """Get number of frames in a DICOM dataset."""
    return int(getattr(ds, 'NumberOfFrames', 1))
    # What are we grabbing frame count?


def extract_frame(pixel_array: np.ndarray, frame_idx: int, num_frames: int) -> np.ndarray:
    """Extract a single frame from pixel array."""
    if num_frames == 1:
        return pixel_array
    return pixel_array[frame_idx]
    # What is the purpose of extracting a single frame?


def normalize_array(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Apply DICOM-standard normalization: Modality LUT -> VOI LUT -> photometric handling."""
    # Keep original dtype for LUT functions, convert to float after
    # Apply Modality LUT (RescaleSlope/Intercept)
    # This converts stored values to output units (e.g., Hounsfield units)
    try:
        arr = apply_modality_lut(arr, ds) # Okay, at this point we take an array apply SOME transformation on it and return ds and arr? arr is image as a numpy array
        # Okay so I looked into this: it applies a modality lookup table or rescale operations to arr, how does it know what to do? it's fed a dataset containing a Modality LUT Module? Where is this coming from?
    except Exception:
        # Fallback: manual application if pydicom function fails
        arr = arr.astype(np.float64)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    # This shouldn't have been a try/catch, either is works or it doesn't. That way we know to manually or rely on pydicom
    # Okay so, inp is arr? we have done what? convert raw scanner values to comparable values. two methods to do this: rescale/intercept -> real_value = (raw_pixel * slope) + intercept BUT where is the slope and intercept #s coming from?
        # Method 2: Lookup Table. I'm assuming dicom image comes with it's own lookup table based off of scanner? I believe so
        # Great analogy is: a universal translator that converts raw scanner gibberish into standardized medical units that mean the same thing across all machines
    
    # Apply VOI LUT (windowing)
    # This maps the output range to display values
    
    try:# This shouldn't have been a try/catch, either is works or it doesn't. That way we know to manually or rely on pydicom

        arr = apply_voi_lut(arr, ds) # this function is choosing what to focus on or value of interest
        # how is this done? window center (WC) = what value should be medium grey?
        # window width (WW) = how wide a range do we display?
        # We're building a bandpass filter around the VOI; for instance say lung is -600 HU, we decide this is our medium grey. We pick a region around it, we stretch that rhange to 0-255, everything else Clip to black (0) or white (255)
        # BUT how is lung chosen? IT's stored in the DICOM metadata, ds.WindowCenter, ds.WindowWidth. The technician and radiologist see these during capture, sometimes there a multiple presets baked in
        # you can literally write it out as: def apply_voi_lut_simplified(arr, center, width): low = center - windth/2; high = center + width/2; arr = np.clip(arr, low, high); return (arr-low)/(high-low)*255
        # Keep in mind; for linear window case only; edgecases include WindowCenter is a list, non-linear LUT, no WindowCenter/Width exists, different VOI LUT functions i.e. linear, sigmoid etc.
    except Exception:
        # Fallback: use window center/width if available
        arr = arr.astype(np.float64)
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            wc = ds.WindowCenter
            ww = ds.WindowWidth
            # Handle multi-valued window settings (use first)
            if isinstance(wc, pydicom.multival.MultiValue):
                wc = wc[0]
            if isinstance(ww, pydicom.multival.MultiValue):
                ww = ww[0]
            wc, ww = float(wc), float(ww)

            # Apply linear windowing
            low = wc - ww / 2
            high = wc + ww / 2
            arr = np.clip(arr, low, high)

    # Convert to float for final normalization
    arr = arr.astype(np.float64)

    # Handle photometric interpretation (MONOCHROME1 = inverted)
    photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
    # Historical reasons; some radiologist prefer film where bone was dark. SO if scans are old-school, we say yo bro 21st is calling. tho tbh, I'm used to this from movies
    if photometric == 'MONOCHROME1':
        arr = arr.max() - arr

    # Normalize to 0-255
    # This is a safety net, after all these LUT transformations,we might end up with
        # 0-1??! rescale to 0-255
        # -500 to +2000? Rescale to 0-255
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min > 1e-6: # this smartness is to prevent division by zero; if every pixel value is the same we get boom division by zero
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    else:
        arr = np.zeros_like(arr)

    # this last catch was to account for: DICOMS without WindowCenter,Width, weird values that don't map, MONOCHROME1 inversion that might shift range
    return arr.astype(np.uint8)


def dicom_to_pil(path: Path, frame_idx: int = 0, size: int = IMAGE_SIZE) -> Optional[Image.Image]:
    """Load DICOM, normalize using standard LUTs, return PIL Image.

    Args:
        path: Path to DICOM file
        frame_idx: Frame index for multi-frame DICOMs # So dicoms are like videos with multiple frames, what I don't know is IF this applies to our mammogramsm would be nice to plot this shit
        size: Output image size (square)

    Returns:
        PIL Image in RGB format, or None if loading fails
    """
    try:
        ds = pydicom.dcmread(str(path)) # ds now contains the entire DICOM dataset; pixel data + metadata

        if not hasattr(ds, 'pixel_array'):
            return None

        pixel_array = ds.pixel_array
        num_frames = get_frame_count(ds)

        # Extract specific frame
        if frame_idx >= num_frames: # IF we asked for frame #50 and only 10 frames exists, return None
            return None
        frame = extract_frame(pixel_array, frame_idx, num_frames)

        # Normalize with LUTs
        arr = normalize_array(frame, ds) # let's work all the magic here; converting raw sensor values, adjusting brightness and constrast, inverse corrections

        # Convert to PIL and resize
        img = Image.fromarray(arr)
        img = img.resize((size, size), Image.LANCZOS) #size = 224x224; Image.LANCZOS a high-quality resizing algorithm. slower but better than nearest neighbors or bilinear

        # Convert to RGB (3-channel for ImageNet pretrained models)
        return img.convert('RGB') # imagenet pretrained models expect RGB, we effectively duplicate values into three channels i.e. Grascale 128 becomes RGB(128,128,128)

    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None


def iter_dicom_frames(path: Path) -> Generator[tuple[int, int], None, None]:
    """Yield (frame_idx, total_frames) for each frame in a DICOM.

    Useful for preprocessing to know how many frames to process.
    """
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        num_frames = get_frame_count(ds)
        for i in range(num_frames):
            yield i, num_frames
    except Exception as e:
        print(f"Warning: Cannot read frame info from {path}: {e}")
        return


def get_dicom_info(path: Path) -> Optional[dict]:
    """Get basic DICOM metadata without loading pixel data."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        return {
            "num_frames": get_frame_count(ds),
            "rows": getattr(ds, 'Rows', None),
            "cols": getattr(ds, 'Columns', None),
            "photometric": getattr(ds, 'PhotometricInterpretation', None),
            "modality": getattr(ds, 'Modality', None),
            "study_description": getattr(ds, 'StudyDescription', None),
            "series_description": getattr(ds, 'SeriesDescription', None),
        }
    except Exception:
        return None


def create_thumbnail(dicom_path: Path, output_path: Path,
                     frame_idx: int = 0, size: int = IMAGE_SIZE) -> bool:
    """Create a thumbnail from a DICOM file.

    Args:
        dicom_path: Source DICOM file
        output_path: Destination PNG path
        frame_idx: Frame to extract (for multi-frame)
        size: Output size

    Returns:
        True if successful, False otherwise
    """
    img = dicom_to_pil(dicom_path, frame_idx, size)
    if img is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), 'PNG')
    return True

from fastcore.foundation import L
from fastcore.xtras import globtastic
from fastcore.basics import Self
from pathlib import Path

def is_dicom_magic(path: Path) -> bool:
    """Check if DICOM magic number at byte 128"""
    try: 
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False

def find_dicoms(root_dir: Path) -> L:
    """Find all DICOMs, return L of (path, study_id) tuples."""
    

def find_dicoms(root_dir: Path) -> Generator[tuple[Path, str], None, None]:
    """Walk directory tree and yield (dicom_path, study_id) tuples.

    Study ID is the immediate parent folder name of the DICOM file.
    """
    root_dir = Path(root_dir)
    for dcm_path in root_dir.rglob("*.dcm"):
        # Study ID is the parent folder name
        study_id = dcm_path.parent.name
        yield dcm_path, study_id

    # Also check for files without .dcm extension (common in DICOM)
    # Try to detect by reading header
    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() != ".dcm":
            try:
                # Quick check for DICOM magic number
                with open(path, 'rb') as f:
                    f.seek(128)
                    if f.read(4) == b'DICM':
                        study_id = path.parent.name
                        yield path, study_id
            except Exception:
                pass
