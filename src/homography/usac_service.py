import cv2, numpy as np, json
from pathlib import Path
from src.matching.summary_writer import save_summary

def run_usac_estimation(feature_dir, out_json="results/homography_usac.json"):
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    src_dir = Path(feature_dir)/"source"
    tgt_dir = Path(feature_dir)/"target"
    results = []
    for sfile in src_dir.glob("*.npz"):
        tfile = tgt_dir / sfile.name.replace("_source", "_target")
        if not tfile.exists(): continue
        sdata, tdata = np.load(sfile), np.load(tfile)
        kp1, kp2 = sdata["keypoints"], tdata["keypoints"]
        n = min(len(kp1), len(kp2), 50)
        idx = np.random.choice(len(kp1), n, replace=False)
        src_pts, dst_pts = kp1[idx], kp2[idx]
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.USAC_ACCURATE)
        if H is None: continue
        src_h = np.hstack([src_pts, np.ones((n, 1))])
        proj = (H @ src_h.T).T; proj /= proj[:, 2:3]
        err = np.mean(np.linalg.norm(proj[:, :2] - dst_pts, axis=1))
        results.append({"file": sfile.name, "method": "USAC", "reprojection_error": float(err)})
    with open(out_json, "w") as f: json.dump(results, f, indent=2)
    save_summary(results, out_json)
    print(f"✅ USAC homography results saved to {out_json}")
