from pathlib import Path

def get_image_pairs(base_dir="data/processed"):
    src_dir = Path(base_dir) / "source"
    tgt_dir = Path(base_dir) / "target"
    pairs = []
    for s in src_dir.glob("*.jpg"):
        t = tgt_dir / s.name
        if t.exists():
            pairs.append((s, t))
    return pairs

def list_feature_files(detector, base_dir="features"):
    base_dir = Path(base_dir) / detector.lower()
    return list((base_dir / "source").glob("*.npz")), list((base_dir / "target").glob("*.npz"))
