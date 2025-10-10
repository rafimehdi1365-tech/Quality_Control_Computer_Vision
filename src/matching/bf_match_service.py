import cv2

def run_bf_match(src_images, tgt_images, detector_name, matcher_name, params=None):
    if params is None:
        params = {}

    # تنظیمات پیش‌فرض
    ratio_thresh = params.get("ratio_test", 0.75)

    # انتخاب detector
    if detector_name == "SIFT":
        detector = cv2.SIFT_create()
    elif detector_name == "ORB":
        detector = cv2.ORB_create()
    elif detector_name == "BRISK":
        detector = cv2.BRISK_create()
    elif detector_name == "AKAZE":
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    # انتخاب matcher
    if matcher_name == "BF":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    elif matcher_name == "FLANN":
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=50)
        bf = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError(f"Unknown matcher: {matcher_name}")

    all_matches = []

    for src, tgt in zip(src_images, tgt_images):
        kp1, des1 = detector.detectAndCompute(src, None)
        kp2, des2 = detector.detectAndCompute(tgt, None)

        if des1 is None or des2 is None:
            all_matches.append([])
            continue

        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe’s ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))

        all_matches.append(good_matches)

    return all_matches
