# src/homography/ransac_service.py
import numpy as np
import cv2
from typing import List, Dict, Optional
from src.homography._helpers import _to_pts_array, reprojection_errors
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

from src.matching.io_utils import append_jsonl, ensure_dir
from pathlib import Path

def run_ransac(matches_list: List[Dict], params: Optional[Dict]=None, save_debug: Optional[Path]=None) -> List[Dict]:
    """
    matches_list: list of dicts produced by matchers (with "pair_id" and "good_matches")
    returns list of homography result dicts
    """
    if params is None:
        params = {}
    reproj_thresh = params.get("ransac_reproj_threshold", 5.0)
    max_iters = params.get("max_iter", 2000)
    confidence = params.get("confidence", 0.995)

    results = []
    if save_debug:
        ensure_dir(Path(save_debug).parent)

    for rec in matches_list:
        pair_id = rec.get("pair_id", "unknown")
        try:
            good = rec.get("good_matches", [])
            if len(good) < 4:
                out = {"pair_id": pair_id, "H": None, "status": "not_enough_points", "n_inliers": 0, "reproj_mean": None, "reproj_median": None, "error": None}
                results.append(out)
                if save_debug:
                    append_jsonl(out, save_debug)
                continue

            src_pts, dst_pts = _to_pts_array(good)
            if src_pts is None:
                out = {"pair_id": pair_id, "H": None, "status": "bad_input", "n_inliers": 0, "reproj_mean": None, "reproj_median": None, "error": "src/dst conversion failed"}
                results.append(out)
                if save_debug:
                    append_jsonl(out, save_debug)
                continue

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh, maxIters=max_iters, confidence=confidence)
            if H is None:
                out = {"pair_id": pair_id, "H": None, "status": "failed", "n_inliers": 0, "reproj_mean": None, "reproj_median": None, "error": "cv_findHomography_failed"}
                results.append(out)
                if save_debug:
                    append_jsonl(out, save_debug)
                continue

            # compute reprojection errors on all points, but mask inliers
            dists = reprojection_errors(H, src_pts, dst_pts)
            if dists is None:
                mean_err = None
                med_err = None
            else:
                mean_err = float(np.mean(dists))
                med_err = float(np.median(dists))

            n_inliers = int(np.sum(mask)) if mask is not None else 0
            out = {"pair_id": pair_id, "H": H.tolist(), "status": "ok", "n_inliers": n_inliers, "reproj_mean": mean_err, "reproj_median": med_err, "error": None}
            results.append(out)
            if save_debug:
                append_jsonl(out, save_debug)

        except Exception as e:
            logger.exception("RANSAC error for %s: %s", pair_id, str(e))
            out = {"pair_id": pair_id, "H": None, "status": "exception", "n_inliers": 0, "reproj_mean": None, "reproj_median": None, "error": str(e)}
            results.append(out)
            if save_debug:
                append_jsonl(out, save_debug)

    return results
