import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_arl_summary(csv_path="results/mewma_arl_summary_by_run.csv"):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10,6))
    plt.barh(df["method"], df["ARL"], color="skyblue")
    plt.xlabel("Average Run Length (ARL)")
    plt.ylabel("Method")
    plt.title("ARL Comparison across Homography Methods")
    plt.tight_layout()
    out_path = Path(csv_path).parent / "arl_comparison.png"
    plt.savefig(out_path, dpi=200)
    print(f"✅ Plot saved to {out_path}")
