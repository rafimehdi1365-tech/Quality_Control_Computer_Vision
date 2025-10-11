# src/matching/bf_match_service.py
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from pathlib import Path

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = __import__("logging").getLogger(__name__)

from src.matching.io_utils import append_jsonl, ensure_dir

def _create_detector(detector_name: str):
    if detector_name.upper() == "SIFT":
        return cv2.SIFT_create()
    if detector_name.upper() == "ORB":
        return cv2.ORB_create()
    if detector_name.upper() == "BRISK":
        return cv2.BRISK_create()
    if detector_name.upper() == "AKAZE":
        return cv2.AKAZE_create()
    raise ValueError(f"Unknown detector: {detector_name}")

def _select_bf_norm(descriptors: np.ndarray):
    # binary descriptors typically uint8 -> HAMMING, float -> L2
    if descriptors is None:
        return cv2.NORM_L2
    if descriptors.dtype == np.uint8:
        return cv2.NORM_HAMMING
    return cv2.NORM_L2

def run_bf_match(
    src_images: List[np.ndarray],
    tgt_images: List[np.ndarray],
    detector_name: str = "SIFT",
    matcher_name: str = "BF",
    params: Optional[Dict] = None,
    save_debug: Optional[Path] = None
) -> List[Dict]:
    """
    Returns list of dicts (one per pair) with good matches and summary.
    Each good match => [x1,y1,x2,y2, distance]
    """
    if params is None:
        params = {}
    ratio = params.get("ratio_test", 0.75)

    detector = _create_detector(detector_name)

    results = []
    save_debug = Path(save_debug) if save_debug is not None else None
    if save_debug:
        ensure_dir(save_debug.parent)

    for idx, (src, tgt) in enumerate(zip(src_images, tgt_images)):
        pair_id = params.get("pair_id_fmt", f"pair_{idx:04d}")
        try:
            if src is None or tgt is None:
                raise ValueError("src or tgt image is None")

            kp1, des1 = detector.detectAndCompute(src, None)
            kp2, des2 = detector.detectAndCompute(tgt, None)

            n1 = 0 if kp1 is None else len(kp1)
            n2 = 0 if kp2 is None else len(kp2)

            if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
                rec = {
                    "pair_id": pair_id,
                    "n_kp1": n1,
                    "n_kp2": n2,
                    "n_raw_matches": 0,
                    "n_good_matches": 0,
                    "good_matches": [],
                    "error": "no_descriptors_or_too_few_keypoints"
                }
                results.append(rec)
                if save_debug:
                    append_jsonl(rec, save_debug)
                continue

            # select matcher
            norm = _select_bf_norm(des1)
            if matcher_name.upper() == "BF":
                matcher = cv2.BFMatcher(norm, crossCheck=False)
            elif matcher_name.upper() == "FLANN":
                # forward to flann but keep here for convenience
                # for binary descriptors use LSH
                if des1.dtype == np.uint8:
                    index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
                else:
                    index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                raise ValueError(f"Unknown matcher: {matcher_name}")

            # knnMatch
            raw_matches = matcher.knnMatch(des1, des2, k=2)
            good = []
            for m in raw_matches:
                # m can be (m,n) or something else; guard
                try:
                    m0, m1 = m
                    if m0.distance < ratio * m1.distance:
                        x1, y1 = kp1[m0.queryIdx].pt
                        x2, y2 = kp2[m0.trainIdx].pt
                        good.append([float(x1), float(y1), float(x2), float(y2), float(m0.distance)])
                except Exception:
                    continue

            rec = {
                "pair_id": pair_id,
                "n_kp1": n1,
                "n_kp2": n2,
                "n_raw_matches": len(raw_matches) if raw_matches is not None else 0,
                "n_good_matches": len(good),
                "good_matches": good,
                "error": None
            }
            results.append(rec)
            if save_debug:
                append_jsonl(rec, save_debug)

        except Exception as e:
            logger.exception("Error matching pair %s: %s", pair_id, str(e))
            rec = {
                "pair_id": pair_id,
                "n_kp1": None,
                "n_kp2": None,
                "n_raw_matches": None,
                "n_good_matches": 0,
                "good_matches": [],
                "error": str(e)
            }
            results.append(rec)
            if save_debug:
                append_jsonl(rec, save_debug)

    return results
