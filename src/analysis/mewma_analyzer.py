import numpy as np, json, matplotlib.pyplot as plt
from pathlib import Path

def compute_mewma(errors, lambda_=0.2):
    z = [errors[0]]
    for i in range(1, len(errors)):
        z_i = lambda_ * errors[i] + (1 - lambda_) * z[-1]
        z.append(z_i)
    return np.array(z)

def run_mewma_plot(input_json, out_dir="results/plots"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(input_json, "r") as f:
        data = json.load(f)
    errs = np.array([d["reprojection_error"] for d in data if d.get("reprojection_error")])
    if len(errs) < 3:
        print(f"⚠️ Not enough data in {input_json}")
        return
    z = compute_mewma(errs)
    plt.figure()
    plt.plot(z, label="MEWMA", color="b")
    plt.axhline(np.mean(z)+2*np.std(z), color='r', linestyle='--', label="UCL")
    plt.axhline(np.mean(z)-2*np.std(z), color='g', linestyle='--', label="LCL")
    plt.title(f"MEWMA Chart: {Path(input_json).stem}")
    plt.legend()
    out_path = Path(out_dir) / f"{Path(input_json).stem}_mewma.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ MEWMA chart saved to {out_path}")
