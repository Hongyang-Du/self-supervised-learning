import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metric(metric, log_files, labels, outpath, ylabel):
    plt.figure(figsize=(8, 6))
    for log_file, label in zip(log_files, labels):
        df = pd.read_csv(log_file)
        plt.plot(df["epoch"], df[metric], label=label, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Comparison")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"✅ Saved {outpath}")

def main():
    logdir = Path("linear_probe_logs")

    # 三个 mask ratio 的日志文件
    log_files = [
        logdir / "mask_0.25_train_log.csv",
        logdir / "mask_0.5_train_log.csv",
        logdir / "mask_0.75_train_log.csv",
    ]
    labels = ["mask_0.25", "mask_0.5", "mask_0.75"]

    # 画 Loss
    plot_metric("train_loss", log_files, labels, logdir / "train_loss_overlay.png", "Train Loss")
    plot_metric("val_loss", log_files, labels, logdir / "val_loss_overlay.png", "Val Loss")

    # 画 Accuracy
    plot_metric("train_acc", log_files, labels, logdir / "train_acc_overlay.png", "Train Accuracy")
    plot_metric("val_acc", log_files, labels, logdir / "val_acc_overlay.png", "Val Accuracy")

if __name__ == "__main__":
    main()