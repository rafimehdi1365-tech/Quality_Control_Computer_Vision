# src/pipelines/method1_pipeline.py
import importlib
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.utils.logger import get_logger
from src.utils.dataset_loader import load_images  # should return (src_images, tgt_images)

logger = get_logger(__name__)


# --------------------
# Utility / SPC helpers
# --------------------
def make_output_dir(base: str, combo_name: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path("results") / f"{ts}" / combo_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: Any, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def simple_mewma_limits(stats: np.ndarray, lamb: float = 0.2, L: float = 3.0):
    """
    Compute MEWMA center and control limits for a 1D statistic series.
    Returns center, sigma_hat, ewma_series, upper_limits, lower_limits
    """
    if len(stats) == 0:
        raise ValueError("Empty stats for MEWMA")

    z = np.zeros(len(stats))
    z[0] = stats[0]
    for i in range(1, len(stats)):
        z[i] = lamb * stats[i] + (1 - lamb) * z[i - 1]
    sigma_hat = np.std(stats, ddof=1)
    # time-varying limit factor
    n = np.arange(1, len(stats) + 1)
    factor = np.sqrt((lamb / (2 - lamb)) * (1 - (1 - lamb) ** (2 * n)))
    upper = (np.mean(stats)) + L * sigma_hat * factor
    lower = (np.mean(stats)) - L * sigma_hat * factor
    return float(np.mean(stats)), float(sigma_hat), z, upper.tolist(), lower.tolist()


def plot_mewma(z: np.ndarray, center: float, upper: List[float], lower: List[float], out_path: Path, title: str):
    plt.figure(figsize=(8, 4))
    plt.plot(z, label="MEWMA stat")
    plt.plot([center] * len(z), "--", label="Center")
    plt.plot(upper, "r--", label="Upper limit")
    plt.plot(lower, "r--", label="Lower limit")
    plt.legend()
    plt.title(title)
    plt.xlabel("sample")
    plt.ylabel("stat")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_arl(mewma_center: float, sigma_hat: float, lamb: float, L: float, gen_next_stat_fn, max_steps=1000):
    """
    Simulate until MEWMA signals, return number of steps until alarm (ARL estimate).
    gen_next_stat_fn: callable that returns next statistic (1d) after shift applied
    """
    z = None
    for i in range(1, max_steps + 1):
        try:
            stat = float(gen_next_stat_fn(i))
        except Exception as e:
            logger.exception("Error generating next stat for ARL simulation")
            raise
        if i == 1:
            z = stat
        else:
            z = lamb * stat + (1 - lamb) * z
        limit = mewma_center + L * sigma_hat * np.sqrt((lamb / (2 - lamb)) * (1 - (1 - lamb) ** (2 * i)))
        if z > limit or z < (mewma_center - (limit - mewma_center)):
            return i  # alarm at step i
    return max_steps


# --------------------
# Dynamic loader helpers
# --------------------
def import_detector_module(name: str):
    module_name = f"src.detectors.{name.lower()}_service"
    try:
        return importlib.import_module(module_name)
    except Exception:
        logger.exception(f"Cannot import detector module {module_name}")
        raise


def import_matcher_module(name: str):
    module_name = f"src.matching.{name.lower()}_match_service"
    try:
        return importlib.import_module(module_name)
    except Exception:
        logger.exception(f"Cannot import matcher module {module_name}")
        raise


def import_homography_module(name: str):
    module_name = f"src.homography.{name.lower()}_service"
    try:
        return importlib.import_module(module_name)
    except Exception:
        logger.exception(f"Cannot import homography module {module_name}")
        raise


# --------------------
# Pipeline
# --------------------
def run_method1_pipeline(
    detector_name: str = "sift",
    matcher_name: str = "bf",
    homography_name: str = "ransac",
    combo_name: Optional[str] = None,
    n_baseline: int = 50,
    n_shifted: int = 50,
    mewma_lambda: float = 0.2,
    L_factor: float = 3.0,
    max_arl_steps: int = 500,
    shift_params: Optional[Dict] = None,
):
    combo_name = combo_name or f"method1__{detector_name.upper()}__{matcher_name.upper()}__{homography_name.upper()}"
    out_dir = make_output_dir("results", combo_name)
    logger.info(f"Output dir: {out_dir}")

    errors = []
    try:
        # 1) load images
        try:
            src_images, tgt_images = load_images()
            logger.info(f"Loaded {len(src_images)} source and {len(tgt_images)} target images")
        except Exception:
            logger.exception("Failed to load images")
            raise

        # 2) import modules
        try:
            detector_mod = import_detector_module(detector_name)
            matcher_mod = import_matcher_module(matcher_name)
            homo_mod = import_homography_module(homography_name)
        except Exception as e:
            errors.append(str(e))
            save_json({"errors": errors}, out_dir / "errors.json")
            raise

        # 3) Baseline sampling loop
        stats_baseline = []
        logger.info("Starting baseline sampling")
        for i in range(min(n_baseline, len(src_images))):
            try:
                s_img = src_images[i]
                t_img = tgt_images[i]
                # detect & describe
                keypoints1, desc1 = detector_mod.detect_and_describe(s_img)
                keypoints2, desc2 = detector_mod.detect_and_describe(t_img)
                # match
                matches = matcher_mod.match_descriptors(desc1, desc2)
                # estimate homography (should return dict with 'H' and 'error' or 'reproj_error')
                hres = homo_mod.estimate_homography(matches, keypoints1, keypoints2)
                # derive one scalar stat per sample (e.g., median reprojection error)
                stat = float(hres.get("reproj_median", hres.get("mean_reproj", hres.get("error", 9999))))
                stats_baseline.append(stat)
            except Exception:
                logger.exception(f"Error in baseline sample {i}")
                errors.append(traceback.format_exc())
        save_json({"baseline_stats": stats_baseline}, out_dir / "baseline_stats.json")

        if len(stats_baseline) < 5:
            raise RuntimeError("Not enough baseline samples to compute limits")

        # 4) compute MEWMA limits
        center, sigma_hat, z_series, upper, lower = simple_mewma_limits(np.array(stats_baseline), lamb=mewma_lambda, L=L_factor)
        save_json(
            {
                "mewma_center": center,
                "sigma_hat": sigma_hat,
                "mewma_z": z_series,
                "upper": upper,
                "lower": lower,
                "params": {"lambda": mewma_lambda, "L": L_factor},
            },
            out_dir / "mewma_limits.json",
        )
        plot_mewma(np.array(z_series), center, upper, lower, out_dir / "mewma_baseline.png", f"{combo_name} MEWMA baseline")

        # 5) Shifted sampling + ARL
        logger.info("Starting shifted sampling and ARL estimation")
        # prepare shift generator
        shift_params = shift_params or {"type": "spatial", "dx": 2.0, "dy": 0.0}
        from src.shift.shift_service import apply_shift  # may raise

        shifted_stats = []

        def gen_shifted_stat(step_index: int):
            # choose one sample (cycle through images)
            idx = (step_index - 1) % len(src_images)
            s_img = src_images[idx]
            t_img = tgt_images[idx]
            # apply shift to target image
            t_img_shifted = apply_shift(t_img, shift_params)
            # re-run detection/matching/homography
            k1, d1 = detector_mod.detect_and_describe(s_img)
            k2, d2 = detector_mod.detect_and_describe(t_img_shifted)
            matches = matcher_mod.match_descriptors(d1, d2)
            hres = homo_mod.estimate_homography(matches, k1, k2)
            stat = float(hres.get("reproj_median", hres.get("mean_reproj", hres.get("error", 9999))))
            shifted_stats.append(stat)
            return stat

        # compute ARL by calling compute_arl with generator
        try:
            arl = compute_arl(center, sigma_hat, mewma_lambda, L_factor, gen_shifted_stat, max_steps=max_arl_steps)
        except Exception:
            logger.exception("Error during ARL computation")
            errors.append(traceback.format_exc())
            arl = None

        save_json({"shift_params": shift_params, "shifted_stats_sample": shifted_stats[:min(50, len(shifted_stats))], "estimated_arl": arl}, out_dir / "shifted_and_arl.json")

        # plot shifted sample mewma (if we have some)
        if len(shifted_stats) > 0:
            _, _, zs, up, low = simple_mewma_limits(np.array(shifted_stats), lamb=mewma_lambda, L=L_factor)
            # compute ewma z for shifted_stats explicitly
            z = np.zeros(len(shifted_stats))
            z[0] = shifted_stats[0]
            for i in range(1, len(shifted_stats)):
                z[i] = mewma_lambda * shifted_stats[i] + (1 - mewma_lambda) * z[i - 1]
            plot_mewma(z, center, up, low, out_dir / "mewma_shifted.png", f"{combo_name} MEWMA shifted")

        # final summary
        summary = {
            "combo": combo_name,
            "n_baseline": len(stats_baseline),
            "baseline_mean": float(np.mean(stats_baseline)) if stats_baseline else None,
            "baseline_std": float(np.std(stats_baseline, ddof=1)) if len(stats_baseline) > 1 else None,
            "arl": arl,
            "errors": errors,
        }
        save_json(summary, out_dir / "summary.json")
        logger.info(f"Pipeline finished for {combo_name}. ARL={arl}")

        return summary

    except Exception as exc:
        logger.exception("Fatal error in method1 pipeline")
        errors.append(traceback.format_exc())
        out_dir = out_dir if 'out_dir' in locals() else make_output_dir("results", "method1_error")
        save_json({"errors": errors}, out_dir / "errors.json")
        raise


if __name__ == "__main__":
    # quick local run example
    res = run_method1_pipeline(detector_name="sift", matcher_name="bf", homography_name="ransac", n_baseline=20, n_shifted=20)
    print(res)
