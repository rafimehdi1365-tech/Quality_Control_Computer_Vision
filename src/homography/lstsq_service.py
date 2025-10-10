import numpy as np, json, cv2
from pathlib import Path
from src.matching.summary_writer import save_summary

def estimate_lstsq(src_pts, dst_pts):
    """Estimate homography via least squares."""
    if len(src_pts) < 4:
        return None
    A, B = [], []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        B.append(u)
        B.append(v)
    A, B = np.array(A), np.array(B)
    h, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    H = np.append(h, 1).reshape(3, 3)
    return H

def reprojection_error(src_pts, dst_pts, H):
    if H is None:
        return np.inf
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    proj = (H @ src_h.T).T
    proj /= proj[:, 2:3]
    return np.mean(np.linalg.norm(proj[:, :2] - dst_pts, axis=1))

def run_lstsq_estimation(feature_dir, out_json="results/homography_lstsq.json"):
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    src_dir = Path(feature_dir)/"source"
    tgt_dir = Path(feature_dir)/"target"
    results = []
    for sfile in src_dir.glob("*.npz"):
        tfile = tgt_dir / sfile.name.replace("_source", "_target")
        if not tfile.exists():
            continue
        sdata, tdata = np.load(sfile), np.load(tfile)
        kp1, kp2 = sdata["keypoints"], tdata["keypoints"]
        n = min(len(kp1), len(kp2), 50)
        idx = np.random.choice(len(kp1), n, replace=False)
        H = estimate_lstsq(kp1[idx], kp2[idx])
        err = reprojection_error(kp1[idx], kp2[idx], H)
        results.append({"file": sfile.name, "method": "LSTSQ", "reprojection_error": float(err)})
    with open(out_json, "w") as f: json.dump(results, f, indent=2)
    save_summary(results, out_json)
    print(f"✅ LSTSQ homography results saved to {out_json}")
def run_lstsq(*args, **kwargs):
    return run_lstsq_estimation(*args, **kwargs)
