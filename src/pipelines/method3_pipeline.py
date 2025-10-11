# src/pipelines/method3_pipeline.py
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


def plot_line(series, out_path: Path, title: str):
    plt.figure(figsize=(8, 4))
    plt.plot(series)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def import_module(prefix: str, name: str):
    try:
        return importlib.import_module(f"{prefix}.{name.lower()}_service")
    except Exception:
        logger.exception(f"Import failure for {prefix}.{name}")
        raise


def run_method3_pipeline(
    detector_name: str = "sift",
    matcher_name: str = "bf",
    homography_name: str = "usac",
    combo_name: Optional[str] = None,
    n_baseline: int = 50,
    mewma_lambda: float = 0.2,
    L_factor: float = 3.0,
    max_arl_steps: int = 500,
    shift_params: Optional[Dict] = None,
    save_keypoints_json: bool = True,
    n_samples: Optional[int] = None,
    params: Optional[Dict] = None,
    **kwargs,
):
    if n_samples is not None:
        logger.warning("Got legacy parameter 'n_samples' — mapping to n_baseline")
        n_baseline = int(n_samples)
    if params:
        logger.info("Received 'params' dict from orchestrator")
        if "save_keypoints_json" in params:
            save_keypoints_json = bool(params["save_keypoints_json"])

    combo_name = combo_name or f"method3__{detector_name.upper()}__{matcher_name.upper()}__{homography_name.upper()}"
    out_dir = make_output_dir("results", combo_name)
    errors = []
    try:
        # load images
        src_images, tgt_images = load_images()
        logger.info(f"Loaded {len(src_images)} src / {len(tgt_images)} tgt images")

        # import modules
        try:
            detector_mod = import_module("src.detectors", detector_name)
            matcher_mod = import_module("src.matching", matcher_name)
            homo_mod = import_module("src.homography", homography_name)
        except Exception:
            errors.append("Import modules failed")
            save_json({"errors": errors}, out_dir / "errors.json")
            raise

        # Step A: Extract & optionally save keypoints (for HTTP transfer later)
        keypoints_store = []
        for i in range(min(n_baseline, len(src_images))):
            try:
                s = src_images[i]
                t = tgt_images[i]
                k1, d1 = detector_mod.detect_and_describe(s)
                k2, d2 = detector_mod.detect_and_describe(t)
                rec = {"idx": i, "kp1_n": len(k1) if k1 is not None else 0, "kp2_n": len(k2) if k2 is not None else 0}
                # Optionally store compact keypoint info
                if save_keypoints_json:
                    # convert keypoints to serializable dict (x,y,angle,response)
                    def kp_to_list(kp):
                        res = []
                        for k in kp:
                            try:
                                res.append({"pt": [float(k.pt[0]), float(k.pt[1])], "size": float(getattr(k, "size", 0.0)), "angle": float(getattr(k, "angle", 0.0))})
                            except Exception:
                                res.append({})
                        return res
                    rec["kp1"] = kp_to_list(k1)
                    rec["kp2"] = kp_to_list(k2)
                keypoints_store.append(rec)
            except Exception:
                logger.exception(f"Error extracting keypoints at i={i}")
                errors.append(traceback.format_exc())
        # save keypoints to JSONL for compact HTTP transfers later
        keypoints_file = out_dir / "keypoints.jsonl"
        with keypoints_file.open("w", encoding="utf-8") as fh:
            for rec in keypoints_store:
                fh.write(json.dumps(rec) + "\n")
        logger.info(f"Saved keypoints jsonl to {keypoints_file}")

        # Step B: Baseline matching/homography stats
        baseline_stats = []
        for i in range(min(n_baseline, len(src_images))):
            try:
                s = src_images[i]
                t = tgt_images[i]
                k1, d1 = detector_mod.detect_and_describe(s)
                k2, d2 = detector_mod.detect_and_describe(t)
                matches = matcher_mod.match_descriptors(d1, d2)
                hres = homo_mod.estimate_homography(matches, k1, k2)
                stat = float(hres.get("reproj_median", hres.get("mean_reproj", hres.get("error", 9999))))
                baseline_stats.append(stat)
            except Exception:
                logger.exception(f"Error computing baseline stat i={i}")
                errors.append(traceback.format_exc())

        save_json({"baseline_stats": baseline_stats}, out_dir / "baseline_stats.json")
        if len(baseline_stats) < 5:
            raise RuntimeError("Insufficient baseline samples for method3")

        center = float(np.mean(baseline_stats))
        sigma_hat = float(np.std(baseline_stats, ddof=1))
        # MEWMA z
        z = np.zeros(len(baseline_stats))
        z[0] = baseline_stats[0]
        for i in range(1, len(baseline_stats)):
            z[i] = mewma_lambda * baseline_stats[i] + (1 - mewma_lambda) * z[i - 1]
        save_json({"center": center, "sigma_hat": sigma_hat, "z": z.tolist()}, out_dir / "mewma.json")
        plot_line(z, out_dir / "mewma_baseline.png", f"{combo_name} MEWMA baseline")

        # Step C: Shift + ARL
        from src.shift.shift_service import apply_shift

        shifted_stats = []

        def gen_shift_stat(i):
            idx = (i - 1) % len(src_images)
            s = src_images[idx]
            t = apply_shift(tgt_images[idx], shift_params or {"type": "spatial", "dx": 4.0, "dy": 0.0})
            k1, d1 = detector_mod.detect_and_describe(s)
            k2, d2 = detector_mod.detect_and_describe(t)
            matches = matcher_mod.match_descriptors(d1, d2)
            hres = homo_mod.estimate_homography(matches, k1, k2)
            stat = float(hres.get("reproj_median", hres.get("mean_reproj", hres.get("error", 9999))))
            shifted_stats.append(stat)
            return stat

        # compute ARL (reuse simple approach)
        def compute_arl_local(center_val, sigma_val, lamb, L, gen_fn, max_steps=500):
            zval = None
            for step in range(1, max_steps + 1):
                try:
                    s = float(gen_fn(step))
                except Exception:
                    logger.exception("Error generating shifted stat for ARL")
                    raise
                if step == 1:
                    zval = s
                else:
                    zval = lamb * s + (1 - lamb) * zval
                limit = center_val + L * sigma_val * np.sqrt((lamb / (2 - lamb)) * (1 - (1 - lamb) ** (2 * step)))
                if zval > limit or zval < (center_val - (limit - center_val)):
                    return step
            return max_steps

        try:
            arl = compute_arl_local(center, sigma_hat, mewma_lambda, L_factor, gen_shift_stat, max_arl_steps)
        except Exception:
            logger.exception("Error computing ARL in method3")
            errors.append(traceback.format_exc())
            arl = None

        save_json({"shift_params": shift_params, "arl": arl, "shifted_preview": shifted_stats[:50]}, out_dir / "shift_arl.json")
        if len(shifted_stats) > 0:
            z_shift = np.zeros(len(shifted_stats))
            z_shift[0] = shifted_stats[0]
            for i in range(1, len(shifted_stats)):
                z_shift[i] = mewma_lambda * shifted_stats[i] + (1 - mewma_lambda) * z_shift[i - 1]
            plot_line(z_shift, out_dir / "mewma_shifted.png", f"{combo_name} MEWMA shifted")

        summary = {"combo": combo_name, "n_baseline": len(baseline_stats), "arl": arl, "errors": errors}
        save_json(summary, out_dir / "summary.json")
        logger.info("Method3 finished")
        return summary

    except Exception:
        logger.exception("Fatal error in method3 pipeline")
        out_dir = out_dir if "out_dir" in locals() else make_output_dir("results", "method3_error")
        save_json({"errors": traceback.format_exc()}, out_dir / "errors.json")
        raise


if __name__ == "__main__":
    res = run_method3_pipeline(detector_name="sift", matcher_name="bf", homography_name="ransac", n_baseline=20)
    print(res)
