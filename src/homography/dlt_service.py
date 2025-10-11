# src/homography/dlt_service.py
from typing import List, Dict, Optional
import numpy as np
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

from src.homography._helpers import H_from_dlt, reprojection_errors
from src.matching.io_utils import append_jsonl, ensure_dir
from pathlib import Path

def run_dlt(matches_list: List[Dict], params: Optional[Dict]=None, save_debug: Optional[Path]=None) -> List[Dict]:
    if params is None: params = {}
    results = []
    if save_debug:
        ensure_dir(Path(save_debug).parent)

    for rec in matches_list:
        pair_id = rec.get("pair_id","unknown")
        try:
            good = rec.get("good_matches", [])
            if len(good) < 4:
                out = {"pair_id":pair_id, "H":None, "status":"not_enough_points", "n_inliers":0, "reproj_mean":None, "reproj_median":None, "error": None}
                results.append(out)
                if save_debug: append_jsonl(out, save_debug)
                continue

            H = H_from_dlt(good)
            if H is None:
                out = {"pair_id":pair_id, "H":None, "status":"failed", "n_inliers":0, "reproj_mean":None, "reproj_median":None, "error":"dlt_failed"}
                results.append(out)
                if save_debug: append_jsonl(out, save_debug)
                continue

            src = np.float32([[m[0],m[1]] for m in good]).reshape(-1,1,2)
            dst = np.float32([[m[2],m[3]] for m in good]).reshape(-1,1,2)
            dists = reprojection_errors(H, src, dst)
            out = {"pair_id":pair_id, "H":H.tolist(), "status":"ok", "n_inliers":len(good), "reproj_mean":float(dists.mean()), "reproj_median":float(np.median(dists)), "error":None}
            results.append(out)
            if save_debug: append_jsonl(out, save_debug)

        except Exception as e:
            logger.exception("DLT error for %s: %s", pair_id, str(e))
            out = {"pair_id":pair_id, "H":None, "status":"exception", "n_inliers":0, "reproj_mean":None, "reproj_median":None, "error":str(e)}
            results.append(out)
            if save_debug: append_jsonl(out, save_debug)

    return results
