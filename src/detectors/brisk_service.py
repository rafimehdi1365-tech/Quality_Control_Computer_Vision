import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

def detect_and_describe(image):
    try:
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brisk = cv2.BRISK_create()
        kps, desc = brisk.detectAndCompute(gray, None)
        if desc is None:
            desc = np.zeros((0, 64), dtype=np.uint8)
        return kps, desc
    except Exception as e:
        logger.exception("BRISK detect_and_describe failed: %s", e)
        raise
