# sort_and_plot_metrics.py

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def extract_epoch_from_ckpt(ckpt_name):
    """
    从 ckpt 名称里提取 epoch 数字。若是 final.pth 或无法解析，就返回 None。
    """
    m = re.match(r"checkpoint_epoch_(\d+)\.pth", ckpt_name)
    if m:
        return int(m.group(1))
    if ckpt_name == "final.pth":
        return None
    return None


def sort_metrics_df(df):
    # 若已有 “epoch” 列且很多条都是 NaN，可考虑覆盖，但这里我们用从 ckpt 名称解析为主
    # 新增一列 epoch_num
    df["epoch_num"] = df["ckpt"].apply(extract_epoch_from_ckpt)

    # 排序：按 exp, 再按 epoch_num，NaN（final）排最后
    df_sorted = df.sort_values(
        by=["exp", "epoch_num"],
        na_position="last"
    ).reset_index(drop=True)
    return df_sorted


def plot_metric_overlay(df, metric, out_path, color_map):
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for exp, grp in df.groupby("exp"):
        g = grp.dropna(subset=[metric])
        # 如果 epoch_num 存在，则排序
        if "epoch_num" in g.columns:
            g = g.sort_values("epoch_num")
            x = g["epoch_num"].tolist()
        else:
            x = list(range(len(g)))
        y = g[metric].tolist()
        if not y:
            continue
        ax.plot(x, y, marker="o", label=exp, color=color_map.get(exp))

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
                        help="原始 metrics_all_epochs.csv 路径")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="保存排序后 CSV 和图像的目录")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df_sorted = sort_metrics_df(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存排序后的 CSV
    sorted_csv_path = out_dir / "metrics_sorted.csv"
    df_sorted.to_csv(sorted_csv_path, index=False)
    print(f"Saved sorted metrics CSV: {sorted_csv_path}")

    # 为每个实验分配颜色
    exps = sorted(df_sorted["exp"].unique())
    default_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    color_map = {exp: default_colors[i % len(default_colors)] for i, exp in enumerate(exps)}

    # 画三张图
    for metric in ["train_loss", "val_loss", "psnr"]:
        if metric not in df_sorted.columns:
            print(f"⚠ skip metric {metric}, not in sorted CSV")
            continue
        out_path = out_dir / f"{metric}_overlay.png"
        plot_metric_overlay(df_sorted, metric, out_path, color_map)

    print("All done.")


if __name__ == "__main__":
    main()