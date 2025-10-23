# src/pipelines/method3_pipeline.py
"""
Robust implementation for method3 pipeline.

Key features:
- tolerant to load_images() returning 2 or 4 items
- defensive checks for detector/matcher/homography interfaces
- error collection per-sample (doesn't abort whole run)
- JSON-safe saving (handles numpy arrays)
- local compute_arl implementation to avoid circular imports
- clear logging for debugging
"""
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


# --------------------
# Utilities
# --------------------
def make_output_dir(base: str, combo_name: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / f"{ts}" / combo_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: Any, path: Path):
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def plot_line(series, out_path: Path, title: str):
    plt.figure(figsize=(8, 4))
    plt.plot(series)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# --------------------
# Robust import helpers (consistent naming to avoid confusion)
# --------------------
def import_detector_module(name: str):
    try:
        return importlib.import_module(f"src.detectors.{name.lower()}_service")
    except Exception:
        logger.exception("Failed to import detector module '%s'", name)
        raise


def import_matcher_module(name: str):
    try:
        return importlib.import_module(f"src.matching.{name.lower()}_match_service")
    except Exception:
        logger.exception("Failed to import matcher module '%s'", name)
        raise


def import_homography_module(name: str):
    try:
        return importlib.import_module(f"src.homography.{name.lower()}_service")
    except Exception:
        logger.exception("Failed to import homography module '%s'", name)
        raise


# --------------------
# load_images wrapper (handles 2 or 4 returns)
# --------------------
def _unpack_load_images():
    res = load_images()
    if isinstance(res, (list, tuple)):
        if len(res) == 2:
            src, tgt = res
            # produce names if dicts
            src_names = list(src.keys()) if isinstance(src, dict) else [str(i) for i in range(len(src))]
            tgt_names = list(tgt.keys()) if isinstance(tgt, dict) else [str(i) for i in range(len(tgt))]
            return src, tgt, src_names, tgt_names
        elif len(res) >= 4:
            return res[0], res[1], res[2], res[3]
    raise RuntimeError("load_images returned unexpected structure; expected 2 or 4 items")


# --------------------
# Small helpers for MEWMA / ARL
# --------------------
def simple_mewma(series: np.ndarray, lamb: float = 0.2) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    if series.size == 0:
        return np.array([])
    z = np.zeros(len(series), dtype=float)
    z[0] = series[0]
    for i in range(1, len(series)):
        z[i] = lamb * series[i] + (1 - lamb) * z[i - 1]
    return z


def compute_arl_local(center_val: float, sigma_val: float, lamb: float, L: float, gen_fn, max_steps: int = 500) -> int:
    z_val = None
    for step in range(1, max_steps + 1):
        stat = float(gen_fn(step))
        if step == 1:
            z_val = stat
        else:
            z_val = lamb * stat + (1 - lamb) * z_val
        limit = center_val + L * sigma_val * np.sqrt((lamb / (2 - lamb)) * (1 - (1 - lamb) ** (2 * step)))
        if z_val > limit or z_val < (center_val - (limit - center_val)):
            return step
    return max_steps


# --------------------
# Method3 pipeline
# --------------------
def run_method3_pipeline(
    detector_name: str = "sift",
    matcher_name: str = "flann",
    homography_name: str = "ransac",
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
    Method3 pipeline:
    - May be used for advanced feature pipelines (multi-scale or preprocessing).
    - Implementation is defensive: missing functions/modules don't abort whole run; sample errors are collected.
    """
    if n_samples is not None:
        logger.warning("Got legacy parameter 'n_samples' — mapping to n_baseline")
        n_baseline = int(n_samples)

    if params:
        logger.info("Received 'params' dict from orchestrator")
        if "shift_params" in params and not shift_params:
            shift_params = params["shift_params"]

    combo_name = combo_name or f"method3__{detector_name.upper()}__{matcher_name.upper()}__{homography_name.upper()}"
    out_dir = make_output_dir("results", combo_name)
    logger.info("Output dir: %s", out_dir)

    errors = []
    try:
        # load images robustly
        src_images, tgt_images, src_names, tgt_names = _unpack_load_images()
        logger.info("Loaded %d src / %d tgt images", len(src_images), len(tgt_images))

        # import modules safely
        detector_mod = import_detector_module(detector_name)
        matcher_mod = import_matcher_module(matcher_name)
        homo_mod = import_homography_module(homography_name)

        # baseline sampling
        baseline_stats = []
        logger.info("Starting baseline sampling (method3)")
        for i in range(min(n_baseline, len(src_images))):
            try:
                s = src_images[i]
                t = tgt_images[i]

                # optional detector preprocess hook
                if hasattr(detector_mod, "preprocess_image"):
                    try:
                        s = detector_mod.preprocess_image(s)
                        t = detector_mod.preprocess_image(t)
                    except Exception:
                        logger.exception("Detector preprocess failed for sample %d; using original images", i)

                # detect & describe
                if not hasattr(detector_mod, "detect_and_describe"):
                    raise RuntimeError(f"Detector module {detector_name} lacks detect_and_describe")

                k1, d1 = detector_mod.detect_and_describe(s)
                k2, d2 = detector_mod.detect_and_describe(t)

                if d1 is None or d2 is None:
                    raise RuntimeError("Descriptors are None")

                # match
                if not hasattr(matcher_mod, "match_descriptors"):
                    raise RuntimeError(f"Matcher module {matcher_name} lacks match_descriptors")
                matches = matcher_mod.match_descriptors(d1, d2)

                # homography estimation
                if not hasattr(homo_mod, "estimate_homography"):
                    raise RuntimeError(f"Homography module {homography_name} lacks estimate_homography")
                hres = homo_mod.estimate_homography(matches, k1, k2)

                stat = float(hres.get("reproj_median", hres.get("mean_reproj", hres.get("error", 9999))))
                baseline_stats.append(stat)

            except Exception:
                logger.exception("Error in baseline sample %d", i)
                errors.append(traceback.format_exc())
                # continue to next sample

        save_json({"baseline_stats": baseline_stats}, out_dir / "baseline_stats.json")

        if len(baseline_stats) < 5:
            raise RuntimeError(f"Not enough baseline samples ({len(baseline_stats)}) to compute limits")

        # compute mewma limits
        center = float(np.mean(baseline_stats))
        sigma_hat = float(np.std(baseline_stats, ddof=1)) if len(baseline_stats) > 1 else 0.0
        z = simple_mewma(np.array(baseline_stats, dtype=float), lamb=mewma_lambda)
        n = len(baseline_stats)
        factors = np.sqrt((mewma_lambda / (2 - mewma_lambda)) * (1 - (1 - mewma_lambda) ** (2 * np.arange(1, n + 1))))
        upper = (center + L_factor * sigma_hat * factors).tolist()
        lower = (center - L_factor * sigma_hat * factors).tolist()

        save_json({"mewma_center": center, "sigma_hat": sigma_hat, "z": z.tolist(), "upper": upper, "lower": lower}, out_dir / "mewma_limits.json")
        plot_line(z, out_dir / "mewma_baseline.png", f"{combo_name} MEWMA baseline")

        # ARL simulation over shifted images
        from src.shift.shift_service import apply_shift

        shifted_stats = []

        def gen_shifted_stat(step_index: int):
            idx = (step_index - 1) % len(src_images)
            s = src_images[idx]
            t = tgt_images[idx]
            try:
                t_shifted = apply_shift(t, shift_params or {"type": "spatial", "dx": 3.0, "dy": 0.0})
            except Exception:
                logger.exception("apply_shift failed at idx %d; using original t", idx)
                t_shifted = t
            try:
                if not hasattr(detector_mod, "detect_and_describe"):
                    raise RuntimeError("Detector missing detect_and_describe")
                k1, d1 = detector_mod.detect_and_describe(s)
                k2, d2 = detector_mod.detect_and_describe(t_shifted)
                if d1 is None or d2 is None:
                    raise RuntimeError("Descriptors None after shift")
                matches = matcher_mod.match_descriptors(d1, d2)
                if not hasattr(homo_mod, "estimate_homography"):
                    raise RuntimeError("Homography module missing estimate_homography")
                hres = homo_mod.estimate_homography(matches, k1, k2)
                stat = float(hres.get("reproj_median", hres.get("mean_reproj", hres.get("error", 9999))))
            except Exception:
                logger.exception("Error computing shifted stat at idx %d", idx)
                stat = 9999.0
            shifted_stats.append(stat)
            return stat

        try:
            arl = compute_arl_local(center, sigma_hat, mewma_lambda, L_factor, gen_shifted_stat, max_steps=max_arl_steps)
        except Exception:
            logger.exception("Error during ARL computation")
            errors.append(traceback.format_exc())
            arl = None

        save_json({"shifted_preview": shifted_stats[:50], "arl": arl}, out_dir / "shift_and_arl.json")
        if len(shifted_stats) > 0:
            z_shift = simple_mewma(np.array(shifted_stats, dtype=float), lamb=mewma_lambda)
            plot_line(z_shift, out_dir / "mewma_shifted.png", f"{combo_name} MEWMA shifted")

        summary = {
            "combo": combo_name,
            "n_baseline": len(baseline_stats),
            "baseline_mean": float(np.mean(baseline_stats)) if baseline_stats else None,
            "baseline_std": float(np.std(baseline_stats, ddof=1)) if len(baseline_stats) > 1 else None,
            "arl": arl,
            "errors": errors,
        }
        save_json(summary, out_dir / "summary.json")
        logger.info("Method3 pipeline finished: ARL=%s", arl)
        return summary

    except Exception:
        logger.exception("Fatal error in method3 pipeline")
        save_json({"errors": traceback.format_exc()}, out_dir / "errors.json")
        raise


if __name__ == "__main__":
    # quick smoke test (short baseline)
    res = run_method3_pipeline(detector_name="sift", matcher_name="flann", homography_name="ransac", n_baseline=10)
    print(res)
