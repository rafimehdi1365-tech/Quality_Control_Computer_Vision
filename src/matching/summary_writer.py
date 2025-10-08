import json
from pathlib import Path
import numpy as np

def save_summary(results, out_json):
    """Generate a human-readable summary file alongside JSON results."""
    out_path = Path(out_json)
    summary_path = out_path.with_suffix(".summary.json")

    if not results:
        summary = {"total_pairs": 0, "mean_matches": 0, "mean_distance": None}
    else:
        n_matches = [r["n_matches"] for r in results]
        dists = [r["avg_distance"] for r in results if r["avg_distance"] is not None]
        summary = {
            "total_pairs": len(results),
            "mean_matches": float(np.mean(n_matches)),
            "mean_distance": float(np.mean(dists)) if dists else None,
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
