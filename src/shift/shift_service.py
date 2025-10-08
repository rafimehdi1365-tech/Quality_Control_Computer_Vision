import cv2
import numpy as np
from pathlib import Path

def apply_shift_to_folder(input_dir, output_dir, dx=10, dy=10):
    """Apply pixel-level shift to simulate defect motion."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for img_path in input_dir.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        cv2.imwrite(str(output_dir / img_path.name), shifted)
    print(f"✅ Shifted images saved in {output_dir}")

if __name__ == "__main__":
    apply_shift_to_folder("data/processed/source", "data/shifted/source", dx=5, dy=8)
