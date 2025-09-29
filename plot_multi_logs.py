# plot_metrics_with_legend.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_overlay_with_legend(df, metric, out_path, color_map):
    """
    df: 包含列 "exp", "epoch", metric
    metric: "train_loss", "val_loss" 或 "psnr"
    color_map: dict, exp -> color
    """
    plt.figure(figsize=(8,5))
    ax = plt.gca()
    for exp, grp in df.groupby("exp"):
        g = grp.dropna(subset=["epoch", metric])
        if g.empty:
            continue
        g = g.sort_values("epoch")
        c = color_map.get(exp, None)
        ax.plot(g["epoch"].tolist(), g[metric].tolist(),
                marker="o", label=f"{exp}", color=c)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} over epochs")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Experiment")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="metrics_all_epochs.csv 路径")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="存输出图的目录")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 你有哪些实验，就在这里定义颜色
    exps = sorted(df["exp"].unique())
    # 比如给三个实验指定三种颜色
    default_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    color_map = {exp: default_colors[i % len(default_colors)] for i, exp in enumerate(exps)}

    for metric in ["train_loss", "val_loss", "psnr"]:
        if metric not in df.columns:
            print(f"⚠ skip {metric}, not in CSV")
            continue
        outfile = out_dir / f"{metric}_compare.png"
        plot_overlay_with_legend(df, metric, outfile, color_map)

if __name__ == "__main__":
    main()