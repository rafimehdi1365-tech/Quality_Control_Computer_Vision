# src/shift/shift_service.py
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_shift(image, shift_params):
    """
    shift_params: dict e.g. {"type":"spatial", "dx": 2.0, "dy": 0.0}
    returns warped image (same shape)
    """
    try:
        stype = shift_params.get("type", "spatial")
        if stype == "spatial":
            dx = float(shift_params.get("dx", 0.0))
            dy = float(shift_params.get("dy", 0.0))
            h, w = image.shape[:2]
            M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            shifted = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            return shifted
        else:
            logger.warning("Unknown shift type '%s' — returning original image", stype)
            return image
    except Exception as e:
        logger.exception("apply_shift failed: %s", e)
        raise
