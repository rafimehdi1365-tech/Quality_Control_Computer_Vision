# src/utils/config_parser.py
from pathlib import Path
import yaml
import itertools
import copy

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_combinations(config):
    """
    Accepts two styles of config:
    1) legacy: config['combos'] is already a list of combos
    2) new: config has keys: methods, detectors, matchers, homographies and defaults.params
    Returns: list of combos, each combo is a dict:
      {
        "run_id": "...",
        "method": "...",
        "detector": "...",
        "matcher": "...",
        "homography": "...",
        "params": { ... }
      }
    """
    if not config:
        return []

    # if user provided explicit combos list, normalize and return
    if "combos" in config and isinstance(config["combos"], list) and len(config["combos"])>0:
        # normalize: ensure params exist for each combo
        combos = []
        for c in config["combos"]:
            combo = dict(c)
            combo.setdefault("params", config.get("defaults", {}).get("params", {}))
            combos.append(combo)
        return combos

    # otherwise expand from lists
    methods = config.get("methods", ["method1"])
    detectors = config.get("detectors", [])
    matchers = config.get("matchers", [])
    homographies = config.get("homographies", [])
    defaults = config.get("defaults", {}).get("params", {})

    combos = []
    for method, det, mat, homo in itertools.product(methods, detectors, matchers, homographies):
        run_id = f"{method}__{det}__{mat}__{homo}"
        combo = {
            "run_id": run_id,
            "method": method,
            "detector": det,
            "matcher": mat,
            "homography": homo,
            "params": copy.deepcopy(defaults)
        }
        combos.append(combo)
    return combos
