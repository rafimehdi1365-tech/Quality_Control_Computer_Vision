import numpy as np, json
from pathlib import Path

def compute_baseline_limits(result_dir="results"):
    homography_files = list(Path(result_dir).glob("homography_*.json"))
    all_errors = []
    for f in homography_files:
        data = json.load(open(f))
        errs = [r["reprojection_error"] for r in data if r.get("reprojection_error")]
        all_errors.extend(errs)
    mu, sigma = np.mean(all_errors), np.std(all_errors)
    limits = {"mean": mu, "ucl": mu + 3*sigma, "lcl": mu - 3*sigma}
    out_path = Path(result_dir) / "baseline_limits.json"
    json.dump(limits, open(out_path, "w"), indent=2)
    print(f"✅ Baseline limits saved to {out_path}")
