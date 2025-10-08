import cv2
import numpy as np
from pathlib import Path

def extract_akaze_features(folder, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    akaze = cv2.AKAZE_create()

    for img_path in sorted(Path(folder).glob("*.jpg")):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kps, desc = akaze.detectAndCompute(img, None)
        if desc is None:
            print(f"⚠️ No AKAZE keypoints in {img_path.name}")
            continue
        np.savez_compressed(out_dir / f"{img_path.stem}_akaze.npz",
                            keypoints=np.array([kp.pt for kp in kps]),
                            descriptors=desc)
    print(f"✅ AKAZE features saved to {out_dir}")

if __name__ == "__main__":
    extract_akaze_features("data/processed/source", "features/akaze/source")
    extract_akaze_features("data/processed/target", "features/akaze/target")
