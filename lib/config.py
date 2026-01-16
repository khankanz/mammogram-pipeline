"""Configuration and constants for the DICOM classification pipeline."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DICOM_DIR = DATA_DIR / "dicom"
THUMBNAIL_DIR = DATA_DIR / "thumbnails"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "labels.db"

# Test data location
TEST_DICOM_DIR = BASE_DIR / "test"

# Image settings
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3  # RGB for ImageNet pretrained

# Training settings
BATCH_SIZE = 32  # Safe for 8GB VRAM
VALID_PCT = 0.2
RANDOM_SEED = 42
FINE_TUNE_EPOCHS = 6

# Data split settings
HOLDOUT_PCT = 0.10
EXPLORATION_RATE = 0.10

# Preprocessing
NUM_WORKERS = 8
PROGRESS_INTERVAL = 1000

# Labeling UI
LABEL_SERVER_PORT = 5005
UNDO_STACK_SIZE = 10
PRELOAD_COUNT = 3

# Inference
INFERENCE_BATCH_SIZE = 32

# Label values
LABEL_NO = 0
LABEL_YES = 1
LABEL_UNLABELED = None

# Ensure directories exist
def ensure_dirs():
    """Create required directories if they don't exist."""
    for d in [DATA_DIR, DICOM_DIR, THUMBNAIL_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
