import cv2
from pathlib import Path

def load_images_from_folder(folder):
    """Load all valid images from folder into dict{name:image_array}."""
    folder = Path(folder)
    imgs = {}
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    for fn in sorted(folder.iterdir()):
        if fn.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            img = cv2.imread(str(fn))
            if img is not None:
                imgs[fn.name] = img
    return imgs

def load_dataset(base_dir="data"):
    """
    Load source and target images.
    Returns:
      src_imgs, tgt_imgs : dicts mapping filename -> image_array
    """
    base = Path(base_dir)
    src_dir = base / "source"
    tgt_dir = base / "target"

    if not src_dir.exists() or not tgt_dir.exists():
        raise FileNotFoundError(f"Expected folders: {src_dir}, {tgt_dir}")

    src_imgs = load_images_from_folder(src_dir)
    tgt_imgs = load_images_from_folder(tgt_dir)

    if len(src_imgs) == 0 or len(tgt_imgs) == 0:
        raise ValueError("No images found in source/ or target/")

    return src_imgs, tgt_imgs
