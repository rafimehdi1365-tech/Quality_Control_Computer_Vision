import cv2
import numpy as np
import os
from pathlib import Path

def autocrop_image(img):
    """Auto-crop image by removing background (based on Otsu threshold)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img  # if no foreground found, return as is
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]
    return cropped

def preprocess_folder(in_dir, out_dir, resize_to=(256, 256)):
    """Apply autocrop, resize and grayscale conversion to all images in a folder."""
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fn in sorted(in_dir.iterdir()):
        if fn.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        img = cv2.imread(str(fn))
        if img is None:
            print(f"⚠️ Failed to read {fn}")
            continue

        img_cropped = autocrop_image(img)
        img_resized = cv2.resize(img_cropped, resize_to)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        out_path = out_dir / fn.name
        cv2.imwrite(str(out_path), gray)
    print(f"✅ Preprocessed images saved in {out_dir}")

def preprocess_all(base_data_dir="data"):
    """Preprocess both source and target folders."""
    src_in = Path(base_data_dir) / "source"
    tgt_in = Path(base_data_dir) / "target"
    src_out = Path(base_data_dir) / "processed/source"
    tgt_out = Path(base_data_dir) / "processed/target"

    preprocess_folder(src_in, src_out)
    preprocess_folder(tgt_in, tgt_out)

if __name__ == "__main__":
    preprocess_all()
