# src/homography/usac_service.py
import numpy as np
import cv2
from typing import Dict, List, Optional
from pathlib import Path

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

from src.homography._helpers import _to_pts_array, reprojection_errors
from src.matching.io_utils import append_jsonl, ensure_dir


def estimate_homography(matches: List, keypoints1, keypoints2, params: Optional[Dict] = None, save_debug: Optional[Path] = None) -> Dict:
    """
    Estimate homography using USAC (Universal Sample Consensus).
    Requires OpenCV built with USAC support.
    """
    if params is None:
        params = {}
    results = {}

    try:
        if len(matches) < 4:
            return {"H": None, "status": "not_enough_points", "n_inliers": 0}

        src_pts, dst_pts = _to_pts_array(matches, keypoints1, keypoints2)
        if src_pts is None:
            return {"H": None, "status": "bad_input", "n_inliers": 0, "error": "src/dst conversion failed"}

        # USAC (OpenCV ≥ 4.5.1)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_DEFAULT, ransacReprojThreshold=5.0)
        if H is None:
            return {"H": None, "status": "failed", "n_inliers": 0, "error": "cv_findHomography_failed"}

        dists = reprojection_errors(H, src_pts, dst_pts)
        results = {
            "H": H.tolist(),
            "status": "ok",
            "n_inliers": int(np.sum(mask)) if mask is not None else 0,
            "reproj_mean": float(np.mean(dists)) if dists is not None else None,
            "reproj_median": float(np.median(dists)) if dists is not None else None,
        }
        if save_debug:
            ensure_dir(Path(save_debug).parent)
            append_jsonl(results, save_debug)

    except Exception as e:
        logger.exception("USAC error: %s", str(e))
        results = {"H": None, "status": "exception", "error": str(e)}

    return results
