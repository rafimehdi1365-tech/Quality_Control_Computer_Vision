from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from save_feature import load_feature_npz  # relative import

app = FastAPI(title="Matcher")

class MatchRequest(BaseModel):
    detector: str
    src: str
    tgt: str
    params: dict = {}

@app.post("/match")
def match(req: MatchRequest):
    # load features
    src_path = f"/data/features/{req.detector}/{req.src}.npz"
    tgt_path = f"/data/features/{req.detector}/{req.tgt}.npz"
    src_kp, src_desc, _ = load_feature_npz(src_path)
    tgt_kp, tgt_desc, _ = load_feature_npz(tgt_path)
    method = req.params.get("method", "BF")  # BF or FLANN
    # BF matcher (simple)
    if method == "BF":
        import cv2
        if src_desc is None or tgt_desc is None:
            return {"matches": [], "n_matches": 0}
        # handle binary vs float
        if src_desc.dtype == np.uint8 or req.detector in ('ORB','BRISK','AKAZE'):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            m = bf.match(src_desc, tgt_desc)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            m = bf.match(src_desc.astype('float32'), tgt_desc.astype('float32'))
        matches = [{"queryIdx": int(x.queryIdx), "trainIdx": int(x.trainIdx), "distance": float(x.distance)} for x in sorted(m, key=lambda x: x.distance)]
        return {"matches": matches, "n_matches": len(matches)}
    elif method == "FLANN":
        # basic FLANN wrapper for float descriptors; for binary need LSH params
        import cv2
        if src_desc.dtype == np.uint8:
            # use LSH
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            m = flann.match(src_desc, tgt_desc)
        else:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            m = flann.match(src_desc.astype('float32'), tgt_desc.astype('float32'))
        matches = [{"queryIdx": int(x.queryIdx), "trainIdx": int(x.trainIdx), "distance": float(x.distance)} for x in sorted(m, key=lambda x: x.distance)]
        return {"matches": matches, "n_matches": len(matches)}
    else:
        return {"matches": [], "n_matches": 0}
