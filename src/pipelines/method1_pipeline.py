def run_method1_pipeline(detector, matcher, estimator, dataset, n_samples=None, **kwargs):
    """
    Pipeline for method1 with detector/matcher/estimator combos.
    Args:
        detector: keypoint detector (e.g., SIFT, ORB)
        matcher: feature matcher (BF/FLANN)
        estimator: homography/transform estimator (RANSAC, LSTSQ, ...)
        dataset: list of image pairs
        n_samples: number of samples to process (None = all)
    """

    # محدود کردن سایز دیتاست برای سرعت در GitHub Actions
    if n_samples is not None:
        dataset = dataset[:n_samples]

    results = []
    for img1, img2 in dataset:
        # Step1: detect keypoints
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)

        # Step2: match
        matches = matcher.match(des1, des2)

        # Step3: estimate transform
        H, mask = estimator(kp1, kp2, matches)

        results.append({
            "n_kp1": len(kp1),
            "n_kp2": len(kp2),
            "n_matches": len(matches),
            "success": H is not None
        })

    return results
