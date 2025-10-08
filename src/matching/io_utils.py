import json
import numpy as np
from pathlib import Path

def load_features(src_path, tgt_path):
    """Load keypoints and descriptors from npz feature files."""
    if not src_path.exists() or not tgt_path.exists():
        return None, None, None, None
    src_data, tgt_data = np.load(src_path), np.load(tgt_path)
    return src_data["keypoints"], src_data["descriptors"], tgt_data["keypoints"], tgt_data["descriptors"]

def save_json(data, path):
    """Save JSON file with safe indent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
