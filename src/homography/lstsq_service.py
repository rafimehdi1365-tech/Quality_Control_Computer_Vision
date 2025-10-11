# src/homography/lstsq_service.py
import numpy as np
from typing import List, Dict, Optional
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

from src.homography._helpers import reprojection_errors
from src.matching.io_utils import append_jsonl, ensure_dir
from pathlib import Path

def run_lstsq(matches_list: List[Dict], params: Optional[Dict]=None, save_debug: Optional[Path]=None) -> List[Dict]:
    if params is None:
        params = {}
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
                if save_debug: append_jsonl(out, save_debug)
                continue

            # Build linear system A h = b (like earlier)
            A = []
            b = []
            for (x1,y1,x2,y2, *_ ) in good:
                A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1])
                A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1])
                b.append(x2)
                b.append(y2)
            A = np.array(A)
            b = np.array(b)

            # Solve via lstsq
            h, *_ = np.linalg.lstsq(A, b, rcond=None)
            H = np.append(h, 1).reshape(3,3)

            # reprojection
            src = np.float32([[m[0],m[1]] for m in good]).reshape(-1,1,2)
            dst = np.float32([[m[2],m[3]] for m in good]).reshape(-1,1,2)
            dists = reprojection_errors(H, src, dst)
            mean_err = float(np.mean(dists))
            med_err = float(np.median(dists))
            out = {"pair_id": pair_id, "H": H.tolist(), "status": "ok", "n_inliers": len(good), "reproj_mean": mean_err, "reproj_median": med_err, "error": None}
            results.append(out)
            if save_debug: append_jsonl(out, save_debug)

        except Exception as e:
            logger.exception("LSTSQ error for %s: %s", pair_id, str(e))
            out = {"pair_id": pair_id, "H": None, "status": "exception", "n_inliers": 0, "reproj_mean": None, "reproj_median": None, "error": str(e)}
            results.append(out)
            if save_debug: append_jsonl(out, save_debug)

    return results
