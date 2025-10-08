import yaml
from pathlib import Path

def load_config(path):
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_combinations(config):
    """Generate grid combinations from YAML (detector, matcher, homography)."""
    detectors = config["detectors"]
    matchers = config["matchers"]
    homographies = config["homographies"]
    combos = []
    for d in detectors:
        for m in matchers:
            for h in homographies:
                combos.append({"detector": d, "matcher": m, "homography": h})
    return combos
