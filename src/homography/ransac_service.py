import cv2
import numpy as np

def run_ransac(matches, method_name="RANSAC", params=None):
    if params is None:
        params = {}

    # تنظیمات پیش‌فرض
    ransac_reproj_threshold = params.get("ransac_reproj_threshold", 5.0)
    max_iter = params.get("max_iter", 2000)
    confidence = params.get("confidence", 0.995)

    all_results = []

    for good_matches in matches:
        if len(good_matches) < 4:
            all_results.append({"homography": None, "status": "not_enough_points"})
            continue

        src_pts = np.float32([m[0] for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([m[1] for m in good_matches]).reshape(-1, 1, 2)

        if method_name == "RANSAC":
            method_flag = cv2.RANSAC
        elif method_name == "LMEDS":
            method_flag = cv2.LMEDS
        elif method_name == "USAC":
            method_flag = cv2.USAC_MAGSAC
        elif method_name == "DLT":
            method_flag = 0  # DLT استاندارد
        elif method_name == "LSTSQ":
            # کمترین مربعات -> استفاده از np.linalg.lstsq
            A = []
            b = []
            for (x1, y1), (x2, y2) in good_matches:
                A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1])
                A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1])
                b.append(x2)
                b.append(y2)
            A = np.array(A)
            b = np.array(b)
            h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            H = np.append(h, 1).reshape(3, 3)
            all_results.append({"homography": H.tolist(), "status": "ok"})
            continue
        else:
            raise ValueError(f"Unknown homography method: {method_name}")

        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            method_flag,
            ransac_reproj_threshold,
            maxIters=max_iter,
            confidence=confidence
        )

        if H is None:
            all_results.append({"homography": None, "status": "failed"})
        else:
            all_results.append({"homography": H.tolist(), "status": "ok"})

    return all_results
