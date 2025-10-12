import cv2
from pathlib import Path

def load_images_from_folder(folder):
    """Load all valid images from folder into (list_images, list_names)."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    imgs, names = [], []
    for fn in sorted(folder.iterdir()):
        if fn.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            img = cv2.imread(str(fn))
            if img is not None:
                imgs.append(img)
                names.append(fn.name)

    return imgs, names


def load_images(base_dir="data"):
    """
    Load source and target images for pipelines.
    Returns:
      src_imgs, tgt_imgs : lists of image arrays
      src_names, tgt_names : lists of filenames (same order)
    """
    base = Path(base_dir)
    src_dir = base / "source"
    tgt_dir = base / "target"

    if not src_dir.exists() or not tgt_dir.exists():
        raise FileNotFoundError(f"Expected folders: {src_dir}, {tgt_dir}")

    src_imgs, src_names = load_images_from_folder(src_dir)
    tgt_imgs, tgt_names = load_images_from_folder(tgt_dir)

    if len(src_imgs) == 0 or len(tgt_imgs) == 0:
        raise ValueError("No images found in source/ or target/")

    return src_imgs, tgt_imgs, src_names, tgt_names
