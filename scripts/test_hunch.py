import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)) 

from lib.db import get_db
from lib.config import DB_PATH

db = get_db(DB_PATH)

# Count by split
print("=== Split Distribution ===")
print("split=None:", db.execute("SELECT COUNT(*) FROM labels WHERE split IS NULL").fetchone()[0])
print("split=train:", db.execute("SELECT COUNT(*) FROM labels WHERE split = 'train'").fetchone()[0])
print("split=val:", db.execute("SELECT COUNT(*) FROM labels WHERE split = 'val'").fetchone()[0])
print("split=test:", db.execute("SELECT COUNT(*) FROM labels WHERE split = 'test'").fetchone()[0])

# Count unlabeled available for labeling
unlabeled = db.execute("""
    SELECT COUNT(*) FROM labels 
    WHERE has_biopsy_tool IS NULL 
    AND has_mag_view IS NULL 
    AND (split IS NULL OR split IN ('train', 'val'))
""").fetchone()[0]
print("=== Available to Label ===")
print("Unlabeled (not hold-out):", unlabeled)

# What's this set to?
from lib.config import HOLDOUT_PCT
print(f"HOLDOUT_PCT = {HOLDOUT_PCT}")