import subprocess
from src.utils.config_parser import load_config, get_combinations
from src.utils.logger import log
from pathlib import Path

def run_step(cmd):
    log(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        log(f"❌ Command failed: {cmd}", "ERROR")

def orchestrate_pipeline(config_path="configs/grid_120.yaml"):
    cfg = load_config(config_path)
    combos = get_combinations(cfg)
    log(f"Loaded {len(combos)} combinations.")

    # Example execution flow:
    run_step("python src/preprocessing/preprocess.py")

    for combo in combos:
        det, match, homo = combo["detector"], combo["matcher"], combo["homography"]
        log(f"▶️ Running {det} + {match} + {homo}")

        run_step(f"python src/detectors/{det.lower()}_service.py")
        run_step(f"python src/matching/{match.lower()}_match_service.py")
        run_step(f"python src/homography/{homo.lower()}_service.py")

    log("Pipeline completed successfully ✅")

if __name__ == "__main__":
    orchestrate_pipeline()
