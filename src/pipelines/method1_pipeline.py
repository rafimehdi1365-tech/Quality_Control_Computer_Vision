import json
from src.utils.logger import get_logger
from src.matching.bf_match_service import run_bf_match
from src.homography.ransac_service import run_ransac
from src.utils.dataset_loader import load_images

logger = get_logger(__name__)

def run_method1(detector, matcher, homography, n_samples=20, shift=None, out_file=None):
    """اجرا برای method1 با استفاده از SIFT, BF, RANSAC"""

    logger.info(f"Running method1 with {detector}, {matcher}, {homography}")

    # بارگذاری داده‌ها (images)
    src_images, tgt_images = load_images()

    # استخراج ویژگی‌ها و محاسبه matching
    matches = run_bf_match(src_images, tgt_images, detector, matcher)

    # محاسبه هموگرافی
    homography_results = run_ransac(matches, homography)

    # ذخیره نتایج در فایل خروجی
    if out_file:
        with open(out_file, 'w') as f:
            json.dump(homography_results, f, indent=2)

    return homography_results
