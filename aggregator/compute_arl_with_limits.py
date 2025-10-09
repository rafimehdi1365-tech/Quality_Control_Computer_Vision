import numpy as np, json
from pathlib import Path

def compute_arl(errors, limits):
    mu, ucl = limits["mean"], limits["ucl"]
    count = 0
    for e in errors:
        count += 1
        if e > ucl:
            return count
    return count

def aggregate_arl(result_dir="results"):
    limits = json.load(open(Path(result_dir)/"baseline_limits.json"))
    result = {}
    for f in Path(result_dir).glob("homography_*.json"):
        data = json.load(open(f))
        errs = [d["reprojection_error"] for d in data if d.get("reprojection_error")]
        result[f.stem] = compute_arl(errs, limits)
    out_path = Path(result_dir)/"mewma_arl_summary_by_run.csv"
    with open(out_path, "w") as fw:
        fw.write("method,ARL\n")
        for k, v in result.items():
            fw.write(f"{k},{v}\n")
    print(f"✅ ARL summary CSV saved to {out_path}")
