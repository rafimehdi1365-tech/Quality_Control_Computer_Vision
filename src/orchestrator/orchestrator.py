# src/orchestrator/orchestrator.py
import argparse
import math
import json
import importlib
from pathlib import Path
from datetime import datetime
from src.utils.config_parser import load_yaml, get_combinations
from src.utils.logger import get_logger
import os

# برگرد به ریشه پروژه
os.chdir(Path(__file__).resolve().parents[2])

logger = get_logger(__name__)

def import_runner_for_method(method):
    """
    Dynamic import of pipeline module.
    module path expected: src.pipelines.<method>_pipeline
    preferred function name: run_pipeline / run_methodX
    """
    module_name = f"src.pipelines.{method}_pipeline"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Cannot import pipeline module '{module_name}': {e}")

    # candidate runner names
    for fname in ("run_pipeline", f"run_{method}_pipeline", f"run_{method}"):
        if hasattr(mod, fname):
            return getattr(mod, fname)

    # fallback: search any run_methodX
    for attr in dir(mod):
        if attr.startswith("run_") and callable(getattr(mod, attr)):
            return getattr(mod, attr)

    raise AttributeError(f"No runnable function found in module {module_name}.")


def run_combo(combo, results_dir, mode="ci"):
    run_id = combo["run_id"]
    method = combo["method"]
    detector = combo["detector"]
    matcher = combo["matcher"]
    homography = combo["homography"]
    params = combo.get("params", {})

    logger.info(f"=== RUN {run_id} (mode={mode}) ===")
    combo_dir = Path(results_dir) / run_id
    combo_dir.mkdir(parents=True, exist_ok=True)

    # import pipeline runner
    runner = import_runner_for_method(method)

    # ⚡ صدا زدن مستقیم pipeline (دیگه out_file و shift لازم نیست)
    try:
        summary = runner(
            detector_name=detector,
            matcher_name=matcher,
            homography_name=homography,
            params=params,
            n_baseline=params.get("baseline_samples", 100),
        )
        # ذخیره summary برای orchestrator
        with open(combo_dir / "orchestrator_summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"=== FINISHED {run_id} ===")
        return combo_dir

    except Exception as e:
        logger.error(f"Error running combo {run_id}: {e}")
        return None


def run_batch(grid_path, batch_id, batches=12, mode="ci"):
    cfg = load_yaml(grid_path)
    combos = get_combinations(cfg)
    total = len(combos)
    batch_id = int(batch_id)
    batches = int(batches)
    batch_size = math.ceil(total / batches)
    start = (batch_id - 1) * batch_size
    end = min(start + batch_size, total)
    batch_combos = combos[start:end]

    logger.info(f"Running batch {batch_id}/{batches}: combos {start}..{end-1} (count={len(batch_combos)})")
    results_dir = Path("results") / f"batch_{batch_id}"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for combo in batch_combos:
        combo_dir = run_combo(combo, results_dir, mode)
        if combo_dir:
            summary.append({"run_id": combo["run_id"], "dir": str(combo_dir)})

    with open(results_dir / f"batch_{batch_id}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Batch {batch_id} done. Results in {results_dir}")


def run_full(grid_path, mode="full"):
    cfg = load_yaml(grid_path)
    combos = get_combinations(cfg)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / ts
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Running full grid: {len(combos)} combos -> results/{ts}")

    for combo in combos:
        run_combo(combo, results_dir, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, help="path to grid yaml")
    parser.add_argument("--batch", type=int, help="batch id (1-based)")
    parser.add_argument("--batches", type=int, default=12, help="total number of batches")
    parser.add_argument("--mode", choices=["ci", "full"], default="ci")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.grid, args.batch, batches=args.batches, mode=args.mode)
    else:
        run_full(args.grid, mode=args.mode)
