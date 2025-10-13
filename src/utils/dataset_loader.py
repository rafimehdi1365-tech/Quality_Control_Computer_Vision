# src/utils/dataset_loader.py
from pathlib import Path
import cv2
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def _load_images_from_folder(folder: Path) -> Tuple[List, List]:
    imgs = []
    names = []
    if not folder.exists():
        logger.error("Folder not found: %s", folder)
        return imgs, names
    for fn in sorted(folder.iterdir()):
        if fn.suffix.lower() in IMG_EXTS:
            img = cv2.imread(str(fn))
            if img is None:
                logger.warning("cv2.imread returned None for %s", fn)
                continue
            imgs.append(img)
            names.append(fn.name)
    return imgs, names

def load_images(base_dir: str = "data"):
    """
    Load source and target images and return:
      src_images (list), tgt_images (list), src_names (list), tgt_names (list)

    The lists are kept in the directory-sorted order. The pipeline expects indexable lists.
    Raises FileNotFoundError or ValueError on serious problems.
    """
    base = Path(base_dir)
    src_dir = base / "source"
    tgt_dir = base / "target"

    if not src_dir.exists() or not tgt_dir.exists():
        raise FileNotFoundError(f"Expected folders: {src_dir}, {tgt_dir}")

    src_imgs, src_names = _load_images_from_folder(src_dir)
    tgt_imgs, tgt_names = _load_images_from_folder(tgt_dir)

    if len(src_imgs) == 0 or len(tgt_imgs) == 0:
        raise ValueError("No images found in source/ or target/")

    logger.info("Loaded %d source and %d target images", len(src_imgs), len(tgt_imgs))
    return src_imgs, tgt_imgs, src_names, tgt_names
