import cv2, json, numpy as np
from pathlib import Path
from src.matching.summary_writer import save_summary

def estimate_homography(src_pts, dst_pts, method="RANSAC"):
    if len(src_pts) < 4:
        return None, None
    if method == "RANSAC":
        return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    elif method == "LMEDS":
        return cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
    else:
        return cv2.findHomography(src_pts, dst_pts, 0)

def reprojection_error(src_pts, dst_pts, H):
    if H is None:
        return np.inf
    src_h = np.concatenate([src_pts, np.ones((len(src_pts), 1))], axis=1)
    proj = (H @ src_h.T).T
    proj /= proj[:, 2:3]
    err = np.linalg.norm(proj[:, :2] - dst_pts, axis=1)
    return np.mean(err)

def run_homography_estimation(match_json, feature_dir, detector, method="RANSAC", out_json="results/homography_ransac.json"):
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(match_json, "r") as f:
        matches_data = json.load(f)

    results = []
    for rec in matches_data:
        if rec["n_matches"] < 4:
            continue
        # load descriptors again for simplicity
        sdata = np.load(Path(feature_dir)/"source"/rec["source_file"])
        tdata = np.load(Path(feature_dir)/"target"/rec["target_file"])
        kp1, kp2 = sdata["keypoints"], tdata["keypoints"]

        # dummy random subset (for demonstration)
        idx = np.random.choice(min(len(kp1), len(kp2)), size=min(20, len(kp1), len(kp2)), replace=False)
        src_pts = np.float32(kp1[idx])
        dst_pts = np.float32(kp2[idx])

        H, _ = estimate_homography(src_pts, dst_pts, method)
        err = reprojection_error(src_pts, dst_pts, H)
        results.append({
            "detector": detector,
            "method": method,
            "source": rec["source_file"],
            "target": rec["target_file"],
            "reprojection_error": float(err),
        })

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    save_summary(results, out_json)
    print(f"✅ Homography ({method}) results saved to {out_json}")

if __name__ == "__main__":
    run_homography_estimation("results/bf_results.json", "features/sift", "SIFT", method="RANSAC")
