# src/homography/ransac_service.py
import numpy as np
import cv2
from typing import List, Dict, Optional
from pathlib import Path
import logging
from src.homography._helpers import _to_pts_array, reprojection_errors
from src.matching.io_utils import append_jsonl, ensure_dir

logger = logging.getLogger(__name__)

def estimate_homography(matches: List[Dict], keypoints1, keypoints2, params: Optional[Dict]=None, save_debug: Optional[Path]=None) -> Dict:
    """
    matches: list of dicts {queryIdx, trainIdx, distance}
    returns dict with keys: H (list) or None, status, reproj_median, mean_reproj, n_inliers, error
    """
    try:
        if params is None:
            params = {}
        reproj_thresh = float(params.get("ransac_reproj_threshold", 5.0))
        max_iters = int(params.get("max_iter", 2000))
        confidence = float(params.get("confidence", 0.995))

        if save_debug:
            ensure_dir(Path(save_debug))

        src_pts, dst_pts = _to_pts_array(matches, keypoints1, keypoints2)
        if src_pts is None or dst_pts is None or len(src_pts) < 4:
            out = {"H": None, "status": "not_enough_points", "n_inliers": 0, "reproj_median": None, "mean_reproj": None, "error": "not_enough_points"}
            if save_debug:
                append_jsonl(out, Path(save_debug))
            return out

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh, maxIters=max_iters, confidence=confidence)
        if H is None:
            out = {"H": None, "status": "failed", "n_inliers": 0, "reproj_median": None, "mean_reproj": None, "error": "cv_findHomography_failed"}
            if save_debug:
                append_jsonl(out, Path(save_debug))
            return out

        dists = reprojection_errors(H, src_pts, dst_pts)
        mean_err = float(np.mean(dists)) if dists is not None else None
        med_err = float(np.median(dists)) if dists is not None else None
        n_inliers = int(np.sum(mask)) if mask is not None else 0

        out = {"H": H.tolist(), "status": "ok", "n_inliers": n_inliers, "reproj_median": med_err, "mean_reproj": mean_err, "error": None}
        if save_debug:
            append_jsonl(out, Path(save_debug))
        return out

    except Exception as e:
        logger.exception("RANSAC estimate_homography failed: %s", e)
        out = {"H": None, "status": "exception", "n_inliers": 0, "reproj_median": None, "mean_reproj": None, "error": str(e)}
        if save_debug:
            try:
                append_jsonl(out, Path(save_debug))
            except Exception:
                logger.exception("Could not append debug jsonl")
        return out
