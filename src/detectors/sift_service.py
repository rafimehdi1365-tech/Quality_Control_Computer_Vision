import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

def detect_and_describe(image):
    try:
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kps, desc = sift.detectAndCompute(gray, None)
        if desc is None:
            # to keep shape predictable
            desc = np.zeros((0, 128), dtype=np.float32)
        return kps, desc
    except Exception as e:
        logger.exception("SIFT detect_and_describe failed: %s", e)
        raise
