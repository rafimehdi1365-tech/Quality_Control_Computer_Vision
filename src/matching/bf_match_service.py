import cv2

def run_bf_match(des1, des2, detector_name="SIFT", cross_check=True):
    """
    Run Brute Force matching between two descriptor sets.
    :param des1: descriptors from image 1
    :param des2: descriptors from image 2
    :param detector_name: name of the feature detector (for choosing norm type)
    :param cross_check: enable crossCheck for strict matching
    :return: sorted list of matches
    """
    if detector_name in ["ORB", "BRISK"]:
        norm_type = cv2.NORM_HAMMING
    else:
        norm_type = cv2.NORM_L2
    cross_check = params.get("crossCheck", False)
    if isinstance(cross_check, str):
        cross_check = cross_check.lower() in ("true", "1", "yes")

    bf = cv2.BFMatcher(norm_type, crossCheck=cross_check)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches
