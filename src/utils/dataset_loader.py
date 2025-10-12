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
    Load source and target ima
