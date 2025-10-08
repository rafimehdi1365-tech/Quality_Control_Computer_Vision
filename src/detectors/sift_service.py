import cv2
import numpy as np
from pathlib import Path

def extract_sift_features(folder, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sift = cv2.SIFT_create()

    for img_path in sorted(Path(folder).glob("*.jpg")):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kps, desc = sift.detectAndCompute(img, None)
        if desc is None:
            print(f"⚠️ No keypoints found for {img_path.name}")
            continue

        np.savez_compressed(out_dir / f"{img_path.stem}_sift.npz",
                            keypoints=np.array([kp.pt for kp in kps]),
                            descriptors=desc)
    print(f"✅ SIFT features saved to {out_dir}")

if __name__ == "__main__":
    extract_sift_features("data/processed/source", "features/sift/source")
    extract_sift_features("data/processed/target", "features/sift/target")
