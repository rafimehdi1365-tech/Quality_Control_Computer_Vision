import numpy as np, json
from pathlib import Path
from src.matching.summary_writer import save_summary

def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    T = np.array([[1/std, 0, -mean[0]/std],
                  [0, 1/std, -mean[1]/std],
                  [0, 0, 1]])
    pts_h = np.hstack([pts, np.ones((len(pts),1))])
    return (T @ pts_h.T).T, T

def estimate_dlt(src_pts, dst_pts):
    if len(src_pts) < 4:
        return None
    src_n, T1 = normalize_points(src_pts)
    dst_n, T2 = normalize_points(dst_pts)
    A = []
    for (x, y, _), (u, v, _) in zip(src_n, dst_n):
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ H @ T1
    return H / H[-1, -1]

def reprojection_error(src_pts, dst_pts, H):
    if H is None:
        return np.inf
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    proj = (H @ src_h.T).T
    proj /= proj[:, 2:3]
    return np.mean(np.linalg.norm(proj[:, :2] - dst_pts, axis=1))

def run_dlt_estimation(feature_dir, out_json="results/homography_dlt.json"):
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
        H = estimate_dlt(kp1[idx], kp2[idx])
        err = reprojection_error(kp1[idx], kp2[idx], H)
        results.append({"file": sfile.name, "method": "DLT", "reprojection_error": float(err)})
    with open(out_json, "w") as f: json.dump(results, f, indent=2)
    save_summary(results, out_json)
    print(f"✅ DLT homography results saved to {out_json}")
