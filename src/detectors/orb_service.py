import cv2
import numpy as np
from pathlib import Path

def extract_orb_features(folder, out_dir, nfeatures=500):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    orb = cv2.ORB_create(nfeatures=nfeatures)

    for img_path in sorted(Path(folder).glob("*.jpg")):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kps, desc = orb.detectAndCompute(img, None)
        if desc is None:
            print(f"⚠️ No ORB keypoints in {img_path.name}")
            continue
        np.savez_compressed(out_dir / f"{img_path.stem}_orb.npz",
                            keypoints=np.array([kp.pt for kp in kps]),
                            descriptors=desc)
    print(f"✅ ORB features saved to {out_dir}")

if __name__ == "__main__":
    extract_orb_features("data/processed/source", "features/orb/source")
    extract_orb_features("data/processed/target", "features/orb/target")
