import cv2
import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)

def _to_output(dmatches):
    out = []
    for m in dmatches:
        # m could be cv2.DMatch
        out.append({"queryIdx": int(m.queryIdx), "trainIdx": int(m.trainIdx), "distance": float(m.distance)})
    return out

def match_descriptors(desc1, desc2, ratio_test=0.75, cross_check=False):
    """
    desc1, desc2: numpy arrays (may be empty with shape (0, dim))
    return: list of matches as dicts
    """
    try:
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            logger.debug("Empty descriptors provided to BF matcher")
            return []

        # select norm depending on descriptor type
        if desc1.dtype == np.uint8:
            norm = cv2.NORM_HAMMING
        else:
            norm = cv2.NORM_L2

        bf = cv2.BFMatcher(norm, crossCheck=cross_check)
        if cross_check:
            matches = bf.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)
            return _to_output(matches)

        # else use knn + ratio test
        knn_matches = bf.knnMatch(desc1, desc2, k=2)
        good = []
        for m_n in knn_matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < ratio_test * n.distance:
                good.append(m)
        good = sorted(good, key=lambda x: x.distance)
        return _to_output(good)
    except Exception as e:
        logger.exception("BF matching failed: %s", e)
        return []
