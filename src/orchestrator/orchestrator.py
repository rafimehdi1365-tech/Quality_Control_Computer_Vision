# src/orchestrator/orchestrator.py
import argparse
import math
import json
import importlib
from pathlib import Path
from datetime import datetime
from src.utils.config_parser import load_yaml, get_combinations
from src.utils.logger import get_logger
from src.aggregator.compute_baseline_limits import compute_baseline_from_results
from aggregator.compute_arl_with_limits import compute_arl_with_limits

logger = get_logger(__name__)

def import_runner_for_method(method):
    """
    dynamic import of pipeline module.
    module path expected: src.pipelines.<method>_pipeline
    preferred function name: run_pipeline
    fallback names: run_method1, run_method2, run_method3
    """
    module_name = f"src.pipelines.{method}_pipeline"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Cannot import pipeline module '{module_name}': {e}")
    # prefer run_pipeline
    for fname in ("run_pipeline", f"run_{method}", "run_method"):
        if hasattr(mod, fname):
            return getattr(mod, fname)
    # fallback: search for any callable in module named run_method1/2/3
    for attr in dir(mod):
        if attr.startswith("run_") and callable(getattr(mod, attr)):
            return getattr(mod, attr)
    raise AttributeError(f"No runnable function found in module {module_name}. Please provide run_pipeline(...)")

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

    # dynamic runner
    runner = import_runner_for_method(method)

    # baseline
    baseline_file = combo_dir / "baseline_results.jsonl"
    logger.info(f"Baseline: {params.get('baseline_samples')} samples")
    runner(detector, matcher, homography,
           n_samples=params.get("baseline_samples", 100),
           shift=None,
           out_file=str(baseline_file))

    # compute baseline limits
    baseline_summary = compute_baseline_from_results(str(baseline_file))
    baseline_json = combo_dir / "baseline_limits.json"
    with open(baseline_json, "w") as f:
        json.dump(baseline_summary, f, indent=2)

    # shifts
    shifts = params.get("shifts", [{"dx":5,"dy":5}])
    repeats = params.get("repeats", 1)
    arl_records = []

    for s in shifts:
        for r in range(repeats):
            run_tag = f"shift_dx{s['dx']}_dy{s['dy']}_r{r+1}"
            shifted_file = combo_dir / f"{run_tag}_results.jsonl"
            logger.info(f"Shift run: {run_tag} — samples={params.get('shifted_samples')}")
            runner(detector, matcher, homography,
                   n_samples=params.get("shifted_samples", 100),
                   shift=s,
                   out_file=str(shifted_file))

            arl_res = compute_arl_with_limits(baseline_limits=str(baseline_json), shifted_results=str(shifted_file))
            arl_res.update({"run_tag": run_tag, "shift": s})
            arl_records.append(arl_res)

    with open(combo_dir / "arl_summary.json", "w") as f:
        json.dump(arl_records, f, indent=2)

    logger.info(f"=== FINISHED {run_id} ===")
    return combo_dir

def run_batch(grid_path, batch_id, batches=12, mode="ci"):
    cfg = load_yaml(grid_path)
    combos = get_combinations(cfg)
    total = len(combos)
    batch_id = int(batch_id)
    batches = int(batches)
    batch_size = math.ceil(total / batches)
    start = (batch_id-1)*batch_size
    end = min(start + batch_size, total)
    batch_combos = combos[start:end]

    logger.info(f"Running batch {batch_id}/{batches}: combos {start}..{end-1} (count={len(batch_combos)})")
    results_dir = Path("results") / f"batch_{batch_id}"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for combo in batch_combos:
        try:
            combo_dir = run_combo(combo, results_dir, mode)
            summary.append({"run_id": combo["run_id"], "dir": str(combo_dir)})
        except Exception as e:
            logger.error(f"Error running combo {combo.get('run_id')}: {e}")

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
        try:
            run_combo(combo, results_dir, mode)
        except Exception as e:
            logger.error(f"Error combo {combo.get('run_id')}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, help="path to grid yaml")
    parser.add_argument("--batch", type=int, help="batch id (1-based)")
    parser.add_argument("--batches", type=int, default=12, help="total number of batches")
    parser.add_argument("--mode", choices=["ci","full"], default="ci")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.grid, args.batch, batches=args.batches, mode=args.mode)
    else:
        run_full(args.grid, mode=args.mode)
