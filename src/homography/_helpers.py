# src/homography/_helpers.py
import numpy as np
import cv2

def _to_pts_array(matches, kp1, kp2):
    """Convert keypoints + matches to numpy arrays"""
    if matches is None or len(matches) == 0:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def reprojection_errors(H, pts1, pts2):
    """Compute reprojection error for homography H"""
    if H is None or pts1 is None or pts2 is None:
        return np.inf
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T
    projected = H @ pts1_h
    projected /= projected[2, :]
    diff = pts2.T - projected[:2, :]
    return float(np.mean(np.linalg.norm(diff, axis=0)))

def H_from_dlt(pts1, pts2):
    """Estimate homography using Direct Linear Transform (DLT)"""
    if pts1 is None or pts2 is None or len(pts1) < 4:
        return None
    H, _ = cv2.findHomography(pts1, pts2, 0)  # 0 = regular DLT
    return H
