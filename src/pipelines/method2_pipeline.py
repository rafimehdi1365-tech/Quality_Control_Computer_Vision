# src/pipelines/method2_pipeline.py
import importlib
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.utils.logger import get_logger
from src.utils.dataset_loader import load_images

logger = get_logger(__name__)


def make_output_dir(base: str, combo_name: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / f"{ts}" / combo_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: Any, path: Path):
    """
    JSON dump with safe default for numpy arrays and other non-serializable types.
    """
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def simple_mewma(series: np.ndarray, lamb: float = 0.2) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    if series.size == 0:
        return np.array([])
    z = np.zeros(len(series), dtype=float)
    z[0] = series[0]
    for i in range(1, len(series)):
        z[i] = lamb * series[i] + (1 - lamb) * z[i - 1]
    return z


def plot_series(series, out_path: Path, title: str):
    plt.figure(figsize=(8, 4))
    plt.plot(series)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# resilient import helpers
def import_detector_module(name: str):
    try:
        return importlib.import_module(f"src.detectors.{name.lower()}_service")
    except Exception:
        logger.exception("Detector import failed for '%s'", name)
        raise


def import_matcher_module(name: str):
    try:
        return importlib.import_module(f"src.matching.{name.lower()}_match_service")
    except Exception:
        logger.exception("Matcher import failed for '%s'", name)
        raise


def import_homography_module(name: str):
    try:
        return importlib.import_module(f"src.homography.{name.lower()}_service")
    except Exception:
        logger.exception("Homography import failed for '%s'", name)
        raise


def _unpack_load_images():
    """
    load_images may return either (src_imgs, tgt_imgs) or (src_imgs, tgt_imgs, src_names, tgt_names).
    Normalize to (src_imgs, tgt_imgs, src_names, tgt_names) where names may be None lists.
    """
    res = load_images()
    if isinstance(res, tuple) or isinstance(res, list):
        if len(res) == 2:
            src, tgt = res
            return src, tgt, list(src.keys()) if isinstance(src, dict) else list(range(len(src))), list(tgt.keys()) if isinstance(tgt, dict) else list(range(len(tgt)))
        elif len(res) >= 4:
            return res[0], res[1], res[2], res[3]
    raise RuntimeError("load_images returned unexpected structure")


def compute_arl_local(center_val: float, sigma_val: float, lamb: float, L: float, gen_fn, max_steps: int = 500) -> int:
    """
    Simulate MEWMA until alarm triggers. Returns steps until alarm (or max_steps).
    """
    z_val = None
    for step in range(1, max_steps + 1):
        try:
            stat = float(gen_fn(step))
        except Exception:
            logger.exception("Error generating stat for ARL at step %d", step)
            raise
        if step == 1:
            z_val = stat
        else:
            z_val = lamb * stat + (1 - lamb) * z_val
        limit = center_val + L * sigma_val * np.sqrt((lamb / (2 - lamb)) * (1 - (1 - lamb) ** (2 * step)))
        if z_val > limit or z_val < (center_val - (limit - center_val)):
            return step
    return max_steps


# method2: preprocessing (autocrop/resize/gray) + (optionally) VAE features then matching/homography
def run_method2_pipeline(
    detector_name: str = "sift",
    matcher_name: str = "flann",
    homography_name: str = "lstsq",
    use_vae: bool = True,
    combo_name: Optional[str] = None,
    n_baseline: int = 50,
    mewma_lambda: float = 0.2,
    L_factor: float = 3.0,
    max_arl_steps: int = 500,
    shift_params: Optional[Dict] = None,
    n_samples: Optional[int] = None,
    params: Optional[Dict] = None,
    **kwargs,
):
    """
    Robust method2 pipeline.
    - Handles different return shapes of load_images
    - Falls back when matcher lacks match_features or detector lacks expected functions
    - Logs errors per-sample without stopping whole run
    """
    if n_samples is not None:
        logger.warning("Got legacy parameter 'n_samples' — mapping to n_baseline")
        n_baseline = int(n_samples)

    if params:
        logger.info("Received 'params' dict from orchestrator")
        if "use_vae" in params:
            use_vae = bool(params["use_vae"])
        if "shift_params" in params and not shift_params:
            shift_params = params["shift_params"]

    combo_name = combo_name or f"method2__{detector_name.upper()}__{matcher_name.upper()}__{homography_name.upper()}__VAE_{use_vae}"
    out_dir = make_output_dir("results", combo_name)
    logger.info("Output dir: %s", out_dir)
    errors = []

    try:
        # load images (robust unpack)
        src_images, tgt_images, src_names, tgt_names = _unpack_load_images()
        logger.info("Loaded images: %d src / %d tgt", len(src_images), len(tgt_images))

        # import modules
        detector_mod = import_detector_module(detector_name)
        matcher_mod = import_matcher_module(matcher_name)
        homo_mod = import_homography_module(homography_name)

        # try import vae if requested
        vae_mod = None
        if use_vae:
            try:
                vae_mod = importlib.import_module("src.vae.vae_service")
                logger.info("VAE module loaded")
            except Exception:
                logger.exception("VAE import failed; continuing without VAE")
                use_vae = False
                vae_mod = None

        baseline_stats = []
        logger.info("Collecting baseline statistics (method2)")

        for i in range(min(n_baseline, len(src_images))):
            try:
                s = src_images[i]
                t = tgt_images[i]
                # optional preprocess hook
                if hasattr(detector_mod, "preprocess_image"):
                    try:
                        s = detector_mod.preprocess_image(s)
                        t = detector_mod.preprocess_image(t)
                    except Exception:
                        logger.exception("Detector preprocess failed for sample %d; using original images", i)

                # if using VAE features
                if use_vae and vae_mod:
                    try:
                        feat_s = vae_mod.encode_image(s)
                        feat_t = vae_mod.encode_image(t)
                    except Exception:
                        logger.exception("VAE encoding failed for sample %d; falling back to keypoint descriptors", i)
                        feat_s = feat_t = None

                    if feat_s is not None and feat_t is not None:
                        # matcher may expose match_features or match_descriptors (fallback)
                        if hasattr(matcher_mod, "match_features"):
                            matches = matcher_mod.match_features(feat_s, feat_t)
                            stat = float(np.median([m.get("distance", np.nan) for m in matches]) if matches else 9999.0)
                        elif hasattr(matcher_mod, "match_descriptors"):
                            # try to convert features to something match_descriptors expects (best-effort)
                            try:
                                matches = matcher_mod.match_descriptors(feat_s, feat_t)
                                stat = float(np.median([m.get("distance", np.nan) for m in matches]) if matches else 9999.0)
                            except Exception:
                                logger.exception("Matcher.match_descriptors failed on VAE features; skipping sample %d", i)
                                raise
                        else:
                            logger.warning("Matcher %s has no match_features or match_descriptors; skipping VAE matching for sample %d", matcher_name, i)
                            raise RuntimeError("No matching interface available for VAE")
                    else:
                        # fallback to keypoints when VAE failed
                        k1 = d1 = k2 = d2 = None
                        if hasattr(detector_mod, "detect_and_describe"):
                            k1, d1 = detector_mod.detect_and_describe(s)
                            k2, d2 = detector_mod.detect_and_describe(t)
                            matches = matcher_mod.match_descriptors(d1, d2)
                            hres = homo_mod.estimate_homography(matches, k1, k2)
                            stat = float(hres.get("reproj_median", hres.get("mean_reproj", hres.get("error", 9999))))
                        else:
                            logger.error("No method to get descriptors for sample %d", i)
                            raise RuntimeError("No descriptor method")
                else:
                    # standard keypoint path
                    if not hasattr(detector_mod, "detect_and_describe"):
                        logger.error("Detector module '%s' has no detect_and_describe", detector_name)
                        raise RuntimeError("Detector missing detect_and_describe")
                    k1, d1 = detector_mod.detect_and_describe(s)
                    k2, d2 = detector_mod.detect_and_describe(t)
                    if d1 is None or d2 is None:
                        logger.warning("Descriptors are None for sample %d; skipping", i)
                        raise RuntimeError("Descriptors are None")
                    matches = matcher_mod.match_descriptors(d1, d2)
                    # homography estimation
                    if not hasattr(homo_mod, "estimate_homography"):
                        logger.error("Homography module '%s' missing estimate_homography", homography_name)
                        raise RuntimeError("Homography module missing interface")
                    hres = homo_mod.estimate_homography(matches, k1, k2)
                    stat = float(hres.get("reproj_median", hres.get("mean_reproj", hres.get("error", 9999))))

                baseline_stats.append(stat)

            except Exception:
                logger.exception("Error sampling baseline i=%d", i)
                errors.append(traceback.format_exc())
                # continue to next sample without aborting

        save_json({"baseline_stats": baseline_stats}, out_dir / "baseline_stats.json")

        if len(baseline_stats) < 5:
            raise RuntimeError(f"Insufficient baseline samples: got {len(baseline_stats)}")

        # compute MEWMA limits and save/plot
        center = float(np.mean(baseline_stats))
        sigma_hat = float(np.std(baseline_stats, ddof=1)) if len(baseline_stats) > 1 else 0.0
        z = simple_mewma(np.array(baseline_stats, dtype=float), lamb=mewma_lambda)
        n = len(baseline_stats)
        factors = np.sqrt((mewma_lambda / (2 - mewma_lambda)) * (1 - (1 - mewma_lambda) ** (2 * np.arange(1, n + 1))))
        upper = (center + L_factor * sigma_hat * factors).tolist()
        lower = (center - L_factor * sigma_hat * factors).tolist()

        save_json({"mewma_center": center, "sigma_hat": sigma_hat, "z": z.tolist(), "upper": upper, "lower": lower}, out_dir / "mewma.json")
        plot_series(z, out_dir / "mewma_baseline.png", f"{combo_name} MEWMA baseline")

        # ARL simulation (shifted images)
        from src.shift.shift_service import apply_shift

        shifted_stats = []

        def gen_shift(i):
            idx = (i - 1) % len(src_images)
            s = src_images[idx]
            t = tgt_images[idx]
            t_shifted = apply_shift(t, shift_params or {"type": "spatial", "dx": 3.0, "dy": 0.0})
            # use same logic as baseline for stat extraction (prefer keypoint path)
            try:
                if not hasattr(detector_mod, "detect_and_describe"):
                    raise RuntimeError("Detector missing detect_and_describe")
                k1, d1 = detector_mod.detect_and_describe(s)
                k2, d2 = detector_mod.detect_and_describe(t_shifted)
                if d1 is None or d2 is None:
                    raise RuntimeError("Descriptors None after shift")
                matches = matcher_mod.match_descriptors(d1, d2)
                if not hasattr(homo_mod, "estimate_homography"):
                    raise RuntimeError("Homography module missing")
                hres = homo_mod.estimate_homography(matches, k1, k2)
                stat = float(hres.get("reproj_median", hres.get("mean_reproj", hres.get("error", 9999))))
            except Exception:
                logger.exception("Error computing shifted stat at idx %d", idx)
                stat = 9999.0
            shifted_stats.append(stat)
            return stat

        try:
            arl = compute_arl_local(center, sigma_hat, mewma_lambda, L_factor, gen_shift, max_steps=max_arl_steps)
        except Exception:
            logger.exception("Error computing ARL")
            errors.append(traceback.format_exc())
            arl = None

        save_json({"shifted_stats_preview": shifted_stats[:50], "arl": arl}, out_dir / "shift_and_arl.json")
        if len(shifted_stats) > 0:
            z_shift = simple_mewma(np.array(shifted_stats, dtype=float), lamb=mewma_lambda)
            plot_series(z_shift, out_dir / "mewma_shifted.png", f"{combo_name} MEWMA shifted")

        summary = {
            "combo": combo_name,
            "n_baseline": len(baseline_stats),
            "baseline_mean": float(np.mean(baseline_stats)) if baseline_stats else None,
            "baseline_std": float(np.std(baseline_stats, ddof=1)) if len(baseline_stats) > 1 else None,
            "arl": arl,
            "errors": errors,
        }
        save_json(summary, out_dir / "summary.json")
        logger.info("Method2 pipeline finished successfully for %s", combo_name)
        return summary

    except Exception:
        logger.exception("Fatal error in method2 pipeline")
        save_json({"errors": traceback.format_exc()}, out_dir / "errors.json")
        raise


if __name__ == "__main__":
    res = run_method2_pipeline(detector_name="sift", matcher_name="flann", homography_name="lstsq", use_vae=False, n_baseline=20)
    print(res)
