import json
from pathlib import Path
import numpy as np

def compute_arl_with_limits(baseline_limits, shifted_results):
    """
    Compute ARL (Average Run Length) for shifted results using baseline limits.

    Parameters
    ----------
    baseline_limits : str
        Path to baseline_limits.json (contains mean, ucl, etc.)
    shifted_results : str
        Path to shifted_results.jsonl or .json (contains reprojection_error values)

    Returns
    -------
    dict
        Dictionary with ARL value and metadata.
    """

    # --- Load baseline limits ---
    limits_path = Path(baseline_limits)
    if not limits_path.exists():
        raise FileNotFoundError(f"Baseline limits file not found: {limits_path}")

    limits = json.load(open(limits_path))
    mu = limits.get("mean", 0)
    ucl = limits.get("ucl", mu + 3 * limits.get("std", 1))

    # --- Load shifted results ---
    shifted_path = Path(shifted_results)
    if not shifted_path.exists():
        raise FileNotFoundError(f"Shifted results file not found: {shifted_path}")

    # Support for .jsonl or .json formats
    errors = []
    if shifted_path.suffix == ".jsonl":
        with open(shifted_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "reprojection_error" in d:
                        errors.append(d["reprojection_error"])
                except json.JSONDecodeError:
                    continue
    else:
        data = json.load(open(shifted_path))
        if isinstance(data, list):
            errors = [d.get("reprojection_error") for d in data if d.get("reprojection_error")]

    if not errors:
        raise ValueError(f"No reprojection_error values found in {shifted_path}")

    # --- Compute ARL ---
    count = 0
    for e in errors:
        count += 1
        if e > ucl:
            break

    # If no point exceeded UCL, ARL = total count
    arl_value = count

    # --- Prepare result ---
    result = {
        "arl": arl_value,
        "n_samples": len(errors),
        "ucl": ucl,
        "mean": mu,
        "file": shifted_path.name
    }

    print(f"✅ ARL computed for {shifted_path.name}: {arl_value}")
    return result
