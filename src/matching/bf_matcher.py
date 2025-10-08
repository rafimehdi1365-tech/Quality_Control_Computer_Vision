import cv2, json, numpy as np
from pathlib import Path
from src.matching.summary_writer import save_summary

def bf_match(desc1, desc2, detector):
    """Perform brute-force matching depending on descriptor type."""
    if desc1 is None or desc2 is None:
        return []
    if detector in ("ORB", "BRISK", "AKAZE"):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def run_bf_matching(src_feature_dir, tgt_feature_dir, detector, out_json="results/bf_results.json"):
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

        matches = bf_match(desc1, desc2, detector)
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
    print(f"✅ BFMatcher results saved to {out_json}")

if __name__ == "__main__":
    run_bf_matching("features/sift/source", "features/sift/target", "SIFT")
