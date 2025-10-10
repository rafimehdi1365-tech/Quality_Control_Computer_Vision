import cv2
import numpy as np

def apply_shift(image, shift_type="translation", shift_params=(5,5)):
    """
    اعمال شیفت روی تصویر برای شبیه‌سازی خطا
    shift_type: نوع شیفت ('translation', 'rotation', 'scaling')
    shift_params: پارامترهای شیفت
    """
    h, w = image.shape[:2]
    if shift_type == "translation":
        dx, dy = shift_params
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(image, M, (w, h))
    elif shift_type == "rotation":
        angle = shift_params
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        return cv2.warpAffine(image, M, (w, h))
    elif shift_type == "scaling":
        scale = shift_params
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        return image
