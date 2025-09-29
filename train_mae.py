import os
import csv
import json
import time
import argparse
from pathlib import Path

import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm, trange

from mae_model import MAE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Stop if val loss does not improve for N epochs")
    return parser.parse_args()


def get_galaxy10_dataloaders(batch_size, num_workers=4):
    ds = load_dataset("matthieulel/galaxy10_decals")
    train_set, val_set = ds["train"], ds["test"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    class Galaxy10Dataset(Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.hf = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.hf)

        def __getitem__(self, idx):
            img = self.hf[idx]["image"]
            if self.transform:
                img = self.transform(img)
            return img

    train_dataset = Galaxy10Dataset(train_set, transform)
    val_dataset = Galaxy10Dataset(val_set, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def save_checkpoint(model, optimizer, scaler, epoch, out_dir, use_amp, final=False):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if use_amp else None,
    }
    name = "final.pth" if final else f"checkpoint_epoch_{epoch}.pth"
    torch.save(ckpt, out_dir / "ckpts" / name)


def train_one_epoch(model, dataloader, optimizer, device, scaler=None, use_amp=False, epoch=0, total_epochs=0):
    model.train()
    running_loss = 0.0
    count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", unit="batch", leave=False)
    for imgs in pbar:
        imgs = imgs.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler and use_amp:
            with autocast():
                loss, _, _ = model(imgs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, _, _ = model(imgs)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        count += 1
        pbar.set_postfix({"loss": f"{(running_loss / count):.4f}"})

    return running_loss / count


@torch.no_grad()
def evaluate(model, dataloader, device, scaler=None, use_amp=False):
    model.eval()
    losses = []
    for imgs in dataloader:
        imgs = imgs.to(device)
        if scaler and use_amp:
            with autocast():
                loss, _, _ = model(imgs)
        else:
            loss, _, _ = model(imgs)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    # === 打印配置 ===
    print("=" * 50)
    print("🚀 Training configuration:")
    for k, v in vars(args).items():
        print(f"{k:>18} : {v}")
    print("=" * 50)

    # === 目录 ===
    exp_name = args.exp_name or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path("runs/mae_pretrain") / exp_name
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    # 保存配置（不依赖外部 logger）
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # === 数据/模型/优化器 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_galaxy10_dataloaders(batch_size=args.batch_size)

    model = MAE(
        img_size=224, patch_size=16, in_chans=3,
        embed_dim=768, depth=12, num_heads=12,
        decoder_dim=512, decoder_depth=8, decoder_heads=16,
        mask_ratio=args.mask_ratio
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler() if args.amp else None

    # === CSV 日志（不依赖外部 logger） ===
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f_csv:
        fieldnames = ["step", "epoch", "train_loss", "val_loss", "lr", "mask_ratio", "wall_time"]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        step = 0
        best_val_loss = float("inf")
        patience_counter = 0
        last_epoch = 0

        for epoch in trange(args.epochs, desc="Training epochs", unit="epoch", leave=False):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device,
                scaler, use_amp=args.amp, epoch=epoch, total_epochs=args.epochs
            )
            val_loss = evaluate(model, val_loader, device, scaler, use_amp=args.amp)

            step += len(train_loader)
            writer.writerow({
                "step": step,
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "mask_ratio": args.mask_ratio,
                "wall_time": time.time()
            })
            f_csv.flush()

            print(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # 保存最好
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, scaler, epoch+1, out_dir, args.amp, final=False)
                torch.save(model.encoder.state_dict(), out_dir / "ckpts/best_encoder.pth")
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                    last_epoch = epoch + 1
                    break

            last_epoch = epoch + 1

    # final ckpt
    save_checkpoint(model, optimizer, scaler, last_epoch, out_dir, args.amp, final=True)

    # 追加一条 ablation 汇总（不依赖外部 JSONL 工具）
    ablation_dir = Path("runs/ablations")
    ablation_dir.mkdir(parents=True, exist_ok=True)
    with open(ablation_dir / "ablation_log.jsonl", "a") as f_ab:
        f_ab.write(json.dumps({
            "exp": exp_name,
            "method": "MAE",
            "mask_ratio": args.mask_ratio,
            "epochs": last_epoch,
            "final_val_loss": float(best_val_loss),
            "encoder_ckpt": str(out_dir / "ckpts/best_encoder.pth")
        }) + "\n")


if __name__ == "__main__":
    main()