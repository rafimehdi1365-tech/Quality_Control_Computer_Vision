# src/homography/lstsq_service.py
import numpy as np
import logging
from typing import List, Dict, Optional
from src.homography._helpers import _to_pts_array, reprojection_errors
from src.matching.io_utils import append_jsonl
from pathlib import Path

logger = logging.getLogger(__name__)

def estimate_homography(matches: List[Dict], keypoints1, keypoints2, params: Optional[Dict]=None, save_debug: Optional[Path]=None) -> Dict:
    """
    Solve for H using least-squares (linearized) when more than 4 points.
    """
    try:
        src_pts, dst_pts = _to_pts_array(matches, keypoints1, keypoints2)
        if src_pts is None or len(src_pts) < 4:
            return {"H": None, "status": "not_enough_points", "n_inliers": 0, "reproj_median": None, "mean_reproj": None, "error": "not_enough_points"}

        src = src_pts.reshape(-1, 2)
        dst = dst_pts.reshape(-1, 2)
        N = src.shape[0]
        A = []
        b = []
        for i in range(N):
            x, y = src[i]
            u, v = dst[i]
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
            b.extend([u, v])
        A = np.array(A)
        b = np.array(b)
        # solve A * h = b  (h is 8 params, last param scale=1)
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            sol = np.linalg.pinv(A) @ b
        h = np.concatenate([sol, [1.0]])
        H = h.reshape(3,3)
        dists = reprojection_errors(H, src_pts, dst_pts)
        mean_err = float(np.mean(dists)) if dists is not None else None
        med_err = float(np.median(dists)) if dists is not None else None
        out = {"H": H.tolist(), "status": "ok", "n_inliers": N, "reproj_median": med_err, "mean_reproj": mean_err, "error": None}
        if save_debug:
            append_jsonl(out, Path(save_debug))
        return out
    except Exception as e:
        logger.exception("LSTSQ estimate_homography failed: %s", e)
        return {"H": None, "status": "exception", "n_inliers": 0, "reproj_median": None, "mean_reproj": None, "error": str(e)}
