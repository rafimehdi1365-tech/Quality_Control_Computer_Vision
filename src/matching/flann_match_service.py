import cv2

def run_flann_match(des1, des2, detector_name="SIFT"):
    """
    Run FLANN-based matching between two descriptor sets.
    :param des1: descriptors from image 1
    :param des2: descriptors from image 2
    :param detector_name: name of the feature detector
    :return: good matches (ratio test)
    """
    if detector_name in ["ORB", "BRISK"]:
        # For binary descriptors
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
    else:
        # For float descriptors (SIFT, AKAZE)
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe’s ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return sorted(good_matches, key=lambda x: x.distance)
