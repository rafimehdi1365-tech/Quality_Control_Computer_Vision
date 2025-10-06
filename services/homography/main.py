from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import cv2
import math

app = FastAPI(title="Homography services")

class HomogRequest(BaseModel):
    src_pts: list  # [[x,y],...]
    dst_pts: list
    method: str = "ransac"
    params: dict = {}

@app.post("/homography")
def compute_homography(req: HomogRequest):
    src = np.array(req.src_pts, dtype='float32')
    dst = np.array(req.dst_pts, dtype='float32')
    method = req.method.lower()
    if src.shape[0] < 4 or dst.shape[0] < 4:
        return {"H": None, "inliers": 0, "reproj_mean": None}
    if method == "ransac":
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, req.params.get("th", 3.0))
    elif method == "lmeds":
        H, mask = cv2.findHomography(src, dst, cv2.LMEDS)
    elif method == "dlt":
        # DLT naive using cv2.findHomography without robust method (direct)
        H, mask = cv2.findHomography(src, dst, 0)
    elif method == "lstsq":
        # simple normalized DLT via SVD (least squares)
        # build A matrix...
        def normalized_dlt(src_pts, dst_pts):
            # implement normalized DLT (short, not fully optimized)
            # This is a simple implementation; for speed rely on OpenCV if possible
            # Fallback to cv2.findHomography with 0 flag:
            return cv2.findHomography(src_pts, dst_pts, 0)
        H, mask = normalized_dlt(src, dst)
    else:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, req.params.get("th", 3.0))
    if H is None:
        return {"H": None, "inliers": 0, "reproj_mean": None}
    mask = mask.ravel() if mask is not None else np.zeros(src.shape[0], dtype=int)
    inliers = int(mask.sum())
    # reprojection errors
    ones = np.hstack([src, np.ones((src.shape[0],1))])
    pr = (H @ ones.T).T
    pr = pr / pr[:,2:3]
    errs = np.linalg.norm(pr[:,:2] - dst, axis=1)
    mean_err = float(errs[mask==1].mean()) if inliers>0 else float(errs.mean())
    return {"H": H.flatten().tolist(), "inliers": int(inliers), "reproj_mean": mean_err, "reproj_list": errs.tolist()}
