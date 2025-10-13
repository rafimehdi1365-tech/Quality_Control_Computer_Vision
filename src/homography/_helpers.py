# src/homography/_helpers.py
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def _to_pts_array(matches, keypoints1, keypoints2):
    """
    matches: list of dicts with queryIdx, trainIdx
    keypoints1, keypoints2: lists of cv2.KeyPoint
    returns: src_pts (N,2), dst_pts (N,2) as float32 or (None, None) on error
    """
    try:
        if not matches:
            return None, None
        src = []
        dst = []
        for m in matches:
            q = int(m["queryIdx"])
            t = int(m["trainIdx"])
            if q < 0 or q >= len(keypoints1) or t < 0 or t >= len(keypoints2):
                continue
            pt1 = keypoints1[q].pt
            pt2 = keypoints2[t].pt
            src.append(pt1)
            dst.append(pt2)
        if len(src) == 0:
            return None, None
        src_pts = np.array(src, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(dst, dtype=np.float32).reshape(-1, 1, 2)
        return src_pts, dst_pts
    except Exception as e:
        logger.exception("Error in _to_pts_array: %s", e)
        return None, None

def reprojection_errors(H, src_pts, dst_pts):
    """
    Compute Euclidean reprojection distances for all point pairs.
    src_pts and dst_pts expected shape: (N,1,2)
    Returns 1D array length N
    """
    try:
        if H is None:
            return None
        src = src_pts.reshape(-1, 2)
        dst = dst_pts.reshape(-1, 2)
        N = src.shape[0]
        src_h = np.concatenate([src, np.ones((N,1))], axis=1)  # (N,3)
        proj = (H @ src_h.T).T  # (N,3)
        proj = proj[:, :2] / proj[:, 2:3]
        dists = np.linalg.norm(proj - dst, axis=1)
        return dists
    except Exception as e:
        logger.exception("reprojection_errors failed: %s", e)
        return None
