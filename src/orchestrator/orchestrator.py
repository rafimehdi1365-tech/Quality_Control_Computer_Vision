import os
import json
import argparse
import math
from pathlib import Path
from datetime import datetime
from src.utils.config_parser import load_yaml
from src.utils.logger import get_logger

# فرض بر این است که pipeline های method1, method2, method3 به صورت تابع import شده‌اند
from src.pipelines.method1_pipeline import run_method1
from src.pipelines.method2_pipeline import run_method2
from src.pipelines.method3_pipeline import run_method3

# tools
from aggregator.compute_baseline_limits import compute_baseline_limits
from aggregator.compute_arl_with_limits import compute_arl_with_limits

logger = get_logger(__name__)

# -----------------------
# Orchestrator
# -----------------------

def run_combo(combo, results_dir, mode="ci"):
    """اجرای یک ترکیب از grid (detector + matcher + homography)"""
    run_id = combo["run_id"]
    method = combo["method"]
    detector = combo["detector"]
    matcher = combo["matcher"]
    homography = combo["homography"]
    params = combo["params"]

    logger.info(f"🚀 Running {run_id} [{mode}]")

    # مسیر خروجی‌ها
    combo_dir = results_dir / run_id
    combo_dir.mkdir(parents=True, exist_ok=True)

    # انتخاب تابع مناسب بر اساس method
    if method == "method1":
        runner = run_method1
    elif method == "method2":
        runner = run_method2
    elif method == "method3":
        runner = run_method3
    else:
        raise ValueError(f"Unknown method: {method}")

    # baseline phase
    logger.info(f"→ Baseline ({params['baseline_samples']} samples)")
    baseline_file = combo_dir / "baseline_results.jsonl"
    runner(detector, matcher, homography,
           n_samples=params["baseline_samples"],
           shift=None,
           out_file=baseline_file)

    # compute baseline limits (mean, sigma, UCL)
    baseline_summary = compute_baseline_limits(baseline_file)
    baseline_json = combo_dir / "baseline_limits.json"
    with open(baseline_json, "w") as f:
        json.dump(baseline_summary, f, indent=2)

    # shift scenarios
    shifts = params.get("shifts", [{"dx": 5, "dy": 5}])
    repeats = params.get("repeats", 3)
    shifted_summary = []

    for sidx, shift in enumerate(shifts):
        for rep in range(repeats):
            run_name = f"shift_dx{shift['dx']}_dy{shift['dy']}_r{rep+1}"
            logger.info(f"→ Shift phase {run_name} ({params['shifted_samples']} samples)")
            shifted_file = combo_dir / f"{run_name}_results.jsonl"

            runner(detector, matcher, homography,
                   n_samples=params["shifted_samples"],
                   shift=shift,
                   out_file=shifted_file)

            # compute ARL for this run
            arl_result = compute_arl_with_limits(
                baseline_limits=baseline_json,
                shifted_results=shifted_file
            )
            arl_result["run_name"] = run_name
            shifted_summary.append(arl_result)

    # ذخیره خلاصه ARL ها
    summary_path = combo_dir / "arl_summary.json"
    with open(summary_path, "w") as f:
        json.dump(shifted_summary, f, indent=2)

    logger.info(f"✅ Finished {run_id}")
    return summary_path


def run_batch(grid_path, batch_id, mode="ci"):
    """اجرای بخشی از ترکیب‌ها (batch mode برای GitHub Actions)"""
    config = load_yaml(grid_path)
    combos = config["combos"]

    # تقسیم‌بندی به batch های کوچکتر
    batch_size = math.ceil(len(combos) / 12)  # فرض 12 job موازی
    start_idx = (batch_id - 1) * batch_size
    end_idx = start_idx + batch_size
    batch_combos = combos[start_idx:end_idx]

    logger.info(f"🏁 Starting batch {batch_id} with {len(batch_combos)} combos")

    results_dir = Path("results") / f"batch_{batch_id}"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for combo in batch_combos:
        try:
            summary_path = run_combo(combo, results_dir, mode)
            all_summaries.append({
                "run_id": combo["run_id"],
                "summary_file": str(summary_path)
            })
        except Exception as e:
            logger.error(f"❌ Error in combo {combo['run_id']}: {e}")

    # ذخیره خلاصه batch
    with open(results_dir / f"batch_{batch_id}_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    logger.info(f"✅ Batch {batch_id} finished.")


def run_full(grid_path):
    """اجرای تمام ترکیب‌ها (برای Colab یا VPS)"""
    config = load_yaml(grid_path)
    combos = config["combos"]
    results_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)

    for combo in combos:
        run_combo(combo, results_dir, mode="full")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, required=True,
                        help="Path to grid YAML file")
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch number (for CI mode)")
    parser.add_argument("--mode", type=str, default="ci",
                        choices=["ci", "full"],
                        help="Mode: ci (GitHub) or full (Colab/VPS)")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.grid, args.batch, args.mode)
    else:
        run_full(args.grid)
