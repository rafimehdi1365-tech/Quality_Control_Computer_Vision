import cv2
import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)

def match_descriptors(desc1, desc2, ratio_test=0.75):
    try:
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            logger.debug("Empty descriptors provided to FLANN matcher")
            return []

        # FLANN params: handle float descriptors (SIFT) and binary -> convert if needed
        if desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
            desc2 = desc2.astype(np.float32)

        index_params = dict(algorithm=1, trees=5)  # KDTree for float
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        knn_matches = flann.knnMatch(desc1, desc2, k=2)
        good = []
        for m_n in knn_matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < ratio_test * n.distance:
                good.append(m)
        good = sorted(good, key=lambda x: x.distance)
        return [{"queryIdx": int(m.queryIdx), "trainIdx": int(m.trainIdx), "distance": float(m.distance)} for m in good]
    except Exception as e:
        logger.exception("FLANN matching failed: %s", e)
        return []
