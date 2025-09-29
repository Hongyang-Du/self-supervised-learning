import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_train_val(csv_path, out_dir):
    # 读 log
    df = pd.read_csv(csv_path)

    # 确保有 val_loss
    if "val_loss" not in df.columns:
        raise ValueError("train_log.csv 里没有 val_loss 列，请确认你运行的是带 validation 的版本。")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========== Loss 曲线 ==========
    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    # ========== Learning rate 曲线 ==========
    if "lr" in df.columns:
        plt.figure(figsize=(8, 6))
        plt.plot(df["epoch"], df["lr"], label="Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.title("Learning Rate Schedule")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_dir / "lr_curve.png")
        plt.close()

    print(f"✅ 保存完成: {out_dir}/loss_curve.png 和 lr_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to train_log.csv")
    parser.add_argument("--out_dir", type=str, default = "plots", help="Directory to save plots")
    args = parser.parse_args()

    plot_train_val(args.csv_path, args.out_dir)