import numpy as np, json
from pathlib import Path

def compute_arl(errors, shift=0.5, lambda_=0.2, limit=1000):
    """Simulate shifted data and compute how fast MEWMA crosses UCL."""
    errs = np.array(errors)
    mu, sigma = np.mean(errs), np.std(errs)
    UCL = mu + 3*sigma
    z = errs[0]
    for t in range(1, limit):
        val = errs[t % len(errs)] + shift*sigma
        z = lambda_ * val + (1 - lambda_) * z
        if z > UCL:
            return t  # smaller = faster detection
    return limit

def compare_arl_across_methods(result_dir="results"):
    paths = list(Path(result_dir).glob("homography_*.json"))
    arl_summary = {}
    for p in paths:
        with open(p, "r") as f: data = json.load(f)
        errs = [d["reprojection_error"] for d in data if d.get("reprojection_error")]
        if not errs: continue
        arl = compute_arl(errs)
        arl_summary[p.stem] = round(arl, 2)
    out_path = Path(result_dir)/"arl_summary.json"
    with open(out_path, "w") as f: json.dump(arl_summary, f, indent=2)
    print(f"✅ ARL summary saved to {out_path}")
    return arl_summary

if __name__ == "__main__":
    compare_arl_across_methods()
