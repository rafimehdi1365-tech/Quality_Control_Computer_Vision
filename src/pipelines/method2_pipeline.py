# src/pipelines/method2_pipeline.py
import importlib
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from src.utils.logger import get_logger
from src.utils.dataset_loader import load_images

logger = get_logger(__name__)


def make_output_dir(base: str, combo_name: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path("results") / f"{ts}" / combo_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: Any, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def simple_mewma(series, lamb=0.2):
    z = np.zeros(len(series))
    z[0] = series[0]
    for i in range(1, len(series)):
        z[i] = lamb * series[i] + (1 - lamb) * z[i - 1]
    return z


def plot_series(series, out_path: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(series)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def import_detector(name: str):
    try:
        return importlib.import_module(f"src.detectors.{name.lower()}_service")
    except Exception:
        logger.exception("Detector import failed")
        raise


def import_matcher(name: str):
    try:
        return importlib.import_module(f"src.matching.{name.lower()}_match_service")
    except Exception:
        logger.exception("Matcher import failed")
        raise


def import_homography(name: str):
    try:
        return importlib.import_module(f"src.homography.{name.lower()}_service")
    except Exception:
        logger.exception("Homography import failed")
        raise


# method2: do preprocessing (autocrop/resize/gray) + (optionally) VAE features then matching/homography
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
    errors = []
    try:
        # load images
        src_images, tgt_images, src_names, tgt_names = load_images()
        logger.info(f"Loaded images: {len(src_images)} / {len(tgt_images)}")
        # import modules
        detector_mod = import_detector(detector_name)
        matcher_mod = import_matcher(matcher_name)
        homo_mod = import_homography(homography_name)
        vae_mod = None
        if use_vae:
            try:
                vae_mod = importlib.import_module("src.vae.vae_service")
            except Exception:
                logger.exception("VAE import failed; continuing without VAE")
                use_vae = False

        baseline_stats = []
        logger.info("Collecting baseline statistics (method2)")
        for i in range(min(n_baseline, len(src_images))):
            try:
                s = src_images[i]
                t = tgt_images[i]
                # preprocessing could be a module; call detector's preprocess if available
                if hasattr(detector_mod, "preprocess_image"):
                    s = detector_mod.preprocess_image(s)
                    t = detector_mod.preprocess_image(t)
                # if using vae, compute latent features and use them as descriptors for matching
                if use_vae and vae_mod:
                    feat_s = vae_mod.encode_image(s)
                    feat_t = vae_mod.encode_image(t)
                    # assume matcher_mod has match_features for VAE features
                    matches = matcher_mod.match_features(feat_s, feat_t)
                    stat = float(np.median([m["distance"] for m in matches]) if matches else 9999.0)
                else:
                    k1, d1 = detector_mod.detect_and_describe(s)
                    k2, d2 = detector_mod.detect_and_describe(t)
                    matches = matcher_mod.match_descriptors(d1, d2)
                    # get homography
                    hres = homo_mod.estimate_homography(matches, k1, k2)
                    stat = float(hres.get("reproj_median", hres.get("mean_reproj", hres.get("error", 9999))))
                baseline_stats.append(stat)
            except Exception:
                logger.exception(f"Error sampling baseline i={i}")
                errors.append(traceback.format_exc())
        save_json({"baseline_stats": baseline_stats}, out_dir / "baseline_stats.json")
        if len(baseline_stats) < 5:
            raise RuntimeError("Insufficient baseline samples")

        # compute MEWMA limits (reuse simple approach)
        center = float(np.mean(baseline_stats))
        sigma_hat = float(np.std(baseline_stats, ddof=1))
        z = simple_mewma(np.array(baseline_stats), lamb=mewma_lambda)
        # compute time-varying limits for plotting
        n = len(baseline_stats)
        factors = np.sqrt((mewma_lambda / (2 - mewma_lambda)) * (1 - (1 - mewma_lambda) ** (2 * np.arange(1, n + 1))))
        upper = (center + L_factor * sigma_hat * factors).tolist()
        lower = (center - L_factor * sigma_hat * factors).tolist()
        save_json({"mewma_center": center, "sigma_hat": sigma_hat, "z": z.tolist(), "upper": upper, "lower": lower}, out_dir / "mewma.json")
        plot_series(z, out_dir / "mewma_baseline.png", f"{combo_name} MEWMA baseline")

        # ARL simulation after applying image shift (spatial)
        from src.shift.shift_service import apply_shift

        shifted_stats = []

        def gen_shift(i):
            idx = (i - 1) % len(src_images)
            s = src_images[idx]
            t = tgt_images[idx]
            t_shifted = apply_shift(t, shift_params or {"type": "spatial", "dx": 3.0, "dy": 0.0})
            if use_vae and vae_mod:
                fs = vae_mod.encode_image(s)
                ft = vae_mod.encode_image(t_shifted)
                matches = matcher_mod.match_features(fs, ft)
                stat = float(np.median([m["distance"] for m in matches]) if matches else 9999.0)
            else:
                k1, d1 = detector_mod.detect_and_describe(s)
                k2, d2 = detector_mod.detect_and_describe(t_shifted)
                matches = matcher_mod.match_descriptors(d1, d2)
                hres = homo_mod.estimate_homography(matches, k1, k2)
                stat = float(hres.get("reproj_median", hres.get("mean_reproj", hres.get("error", 9999))))
            shifted_stats.append(stat)
            return stat

        # compute ARL (same compute_arl function as in method1 but reimplemented here to avoid imports)
        def compute_arl_local(center_val, sigma_val, lamb, L, gen_fn, max_steps=500):
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

        try:
            arl = compute_arl_local(center, sigma_hat, mewma_lambda, L_factor, gen_shift, max_steps)
        except Exception:
            logger.exception("Error computing ARL")
            errors.append(traceback.format_exc())
            arl = None

        save_json({"shifted_stats_preview": shifted_stats[:50], "arl": arl}, out_dir / "shift_and_arl.json")
        if len(shifted_stats) > 0:
            z_shift = simple_mewma(np.array(shifted_stats), lamb=mewma_lambda)
            plot_series(z_shift, out_dir / "mewma_shifted.png", f"{combo_name} MEWMA shifted")

        summary = {
            "combo": combo_name,
            "n_baseline": len(baseline_stats),
            "arl": arl,
            "errors": errors,
        }
        save_json(summary, out_dir / "summary.json")
        logger.info("Method2 pipeline finished")
        return summary

    except Exception:
        logger.exception("Fatal error in method2 pipeline")
        save_json({"errors": traceback.format_exc()}, out_dir / "errors.json")
        raise


if __name__ == "__main__":
    res = run_method2_pipeline(detector_name="sift", matcher_name="flann", homography_name="lstsq", use_vae=False, n_baseline=20)
    print(res)
