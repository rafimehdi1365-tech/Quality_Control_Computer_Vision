import cv2, json, numpy as np
from pathlib import Path
from src.matching.summary_writer import save_summary

def flann_match(desc1, desc2, detector):
    if desc1 is None or desc2 is None:
        return []
    if detector in ("ORB", "BRISK", "AKAZE"):
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
    else:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.match(desc1.astype(np.float32), desc2.astype(np.float32))
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def run_flann_matching(src_feature_dir, tgt_feature_dir, detector, out_json="results/flann_results.json"):
    src_dir = Path(src_feature_dir)
    tgt_dir = Path(tgt_feature_dir)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)

    results = []
    for sfile in sorted(src_dir.glob("*.npz")):
        tfile = tgt_dir / sfile.name.replace("_source", "_target")
        if not tfile.exists():
            continue
        sdata = np.load(sfile)
        tdata = np.load(tfile)
        desc1, desc2 = sdata["descriptors"], tdata["descriptors"]

        matches = flann_match(desc1, desc2, detector)
        record = {
            "detector": detector,
            "source_file": sfile.name,
            "target_file": tfile.name,
            "n_matches": len(matches),
            "avg_distance": float(np.mean([m.distance for m in matches])) if matches else None,
        }
        results.append(record)

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    save_summary(results, out_json)
    print(f"✅ FLANN results saved to {out_json}")

if __name__ == "__main__":
    run_flann_matching("features/orb/source", "features/orb/target", "ORB")
