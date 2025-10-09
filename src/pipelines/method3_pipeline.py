from src.utils.logger import get_logger
from src.utils.dataset_loader import load_dataset
from src.shift.shift_service import apply_shift
from src.matching.io_utils import save_jsonl_record
from src.detectors import sift_service, orb_service, brisk_service, akaze_service
from src.matching import bf_match_service, flann_match_service
from src.homography import ransac_service, lstsq_service, dlt_service, lmeds_service, usac_service
import gc

logger = get_logger(__name__)

def run_pipeline(detector, matcher, homography, n_samples, shift, out_file):
    logger.info(f"[Method3] {detector} + {matcher} + {homography}")
    src_imgs, tgt_imgs = load_dataset()
    if shift:
        tgt_imgs = apply_shift(tgt_imgs, dx=shift["dx"], dy=shift["dy"])

    det_mod = {
        "SIFT": sift_service,
        "ORB": orb_service,
        "BRISK": brisk_service,
        "AKAZE": akaze_service,
    }[detector]

    match_mod = {
        "BF": bf_match_service,
        "FLANN": flann_match_service,
    }[matcher]

    homo_mod = {
        "RANSAC": ransac_service,
        "LSTSQ": lstsq_service,
        "DLT": dlt_service,
        "LMEDS": lmeds_service,
        "USAC": usac_service,
    }[homography]

    for sname, simg in src_imgs.items():
        for tname, timg in tgt_imgs.items():
            kp1, desc1 = det_mod.extract(simg)
            kp2, desc2 = det_mod.extract(timg)
            matches = match_mod.match(desc1, desc2)
            if len(matches) < 4:
                continue
            H, err = homo_mod.estimate(kp1, kp2, matches)
            rec = {
                "src": sname,
                "tgt": tname,
                "method": "method3",
                "detector": detector,
                "matcher": matcher,
                "homography": homography,
                "shift": shift,
                "mean_error": err,
            }
            save_jsonl_record(rec, out_file)
            gc.collect()
    logger.info(f"[Method3] Done → {out_file}")
