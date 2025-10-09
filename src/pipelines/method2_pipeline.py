import json
from src.utils.logger import get_logger
from src.matching.flann_match_service import run_flann_match
from src.homography.lstsq_service import run_lstsq
from src.utils.dataset_loader import load_images

logger = get_logger(__name__)

def run_method2(detector, matcher, homography, n_samples=20, shift=None, out_file=None):
    """اجرا برای method2 با استفاده از ORB, FLANN, LSTSQ"""

    logger.info(f"Running method2 with {detector}, {matcher}, {homography}")

    # بارگذاری داده‌ها (images)
    src_images, tgt_images = load_images()

    # استخراج ویژگی‌ها و محاسبه matching
    matches = run_flann_match(src_images, tgt_images, detector, matcher)

    # محاسبه هموگرافی
    homography_results = run_lstsq(matches, homography)

    # ذخیره نتایج در فایل خروجی
    if out_file:
        with open(out_file, 'w') as f:
            json.dump(homography_results, f, indent=2)

    return homography_results
