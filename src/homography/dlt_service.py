# src/homography/dlt_service.py
import numpy as np
import logging
from typing import List, Dict, Optional
from src.homography._helpers import _to_pts_array, reprojection_errors
from src.matching.io_utils import append_jsonl
from pathlib import Path

logger = logging.getLogger(__name__)

def H_from_dlt(src_pts, dst_pts):
    # src_pts dst_pts: shape (N,1,2)
    src = src_pts.reshape(-1, 2)
    dst = dst_pts.reshape(-1, 2)
    N = src.shape[0]
    A = []
    for i in range(N):
        x, y = src[i]
        u, v = dst[i]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3,3)
    return H / H[2,2]

def estimate_homography(matches: List[Dict], keypoints1, keypoints2, params: Optional[Dict]=None, save_debug: Optional[Path]=None) -> Dict:
    try:
        src_pts, dst_pts = _to_pts_array(matches, keypoints1, keypoints2)
        if src_pts is None or len(src_pts) < 4:
            return {"H": None, "status": "not_enough_points", "n_inliers": 0, "reproj_median": None, "mean_reproj": None, "error": "not_enough_points"}
        H = H_from_dlt(src_pts, dst_pts)
        dists = reprojection_errors(H, src_pts, dst_pts)
        mean_err = float(np.mean(dists)) if dists is not None else None
        med_err = float(np.median(dists)) if dists is not None else None
        out = {"H": H.tolist(), "status": "ok", "n_inliers": len(src_pts), "reproj_median": med_err, "mean_reproj": mean_err, "error": None}
        if save_debug:
            append_jsonl(out, Path(save_debug))
        return out
    except Exception as e:
        logger.exception("DLT estimate_homography failed: %s", e)
        return {"H": None, "status": "exception", "n_inliers": 0, "reproj_median": None, "mean_reproj": None, "error": str(e)}
