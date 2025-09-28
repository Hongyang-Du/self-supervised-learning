import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset

import argparse
import time
from pathlib import Path
from tqdm import tqdm
from tqdm import trange

from mae_model import MAE  # 你的模型文件
from utils.logger import CSVLogger, save_config, JSONLAggregator


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
    return parser.parse_args()


def get_galaxy10_dataloader(batch_size, num_workers=4):
    ds = load_dataset("matthieulel/galaxy10_decals")
    split = ds["train"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.2, 0.2, 0.2]),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    class Galaxy10SelfSupervisedDataset(Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.hf = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.hf)

        def __getitem__(self, idx):
            item = self.hf[idx]
            img = item["image"]   # PIL image
            if self.transform:
                img = self.transform(img)
            return img

    dataset = Galaxy10SelfSupervisedDataset(split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            drop_last=True)
    return dataloader


def save_checkpoint(model, optimizer, scaler, epoch, out_dir, args, final=False):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if args.amp else None,
        "args": vars(args)
    }
    name = "final.pth" if final else f"checkpoint_epoch_{epoch}.pth"
    torch.save(ckpt, out_dir / "ckpts" / name)


def load_checkpoint(model, optimizer, scaler, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler and checkpoint["scaler_state"]:
        scaler.load_state_dict(checkpoint["scaler_state"])
    start_epoch = checkpoint["epoch"]
    return model, optimizer, scaler, start_epoch


def train_one_epoch(model, dataloader, optimizer, device, scaler=None, use_amp=False, epoch=0, total_epochs=0):
    model.train()
    running_loss = 0.0
    count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", unit="batch")
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
        avg_loss = running_loss / count

        # 在进度条上显示 loss
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return running_loss / count


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    exp_name = args.exp_name or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path("runs/mae_pretrain") / exp_name
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    # ==== 数据/模型/优化器 ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_galaxy10_dataloader(batch_size=args.batch_size)

    model = MAE(
        img_size=224, patch_size=16, in_chans=3,
        embed_dim=768, depth=12, num_heads=12,
        decoder_dim=512, decoder_depth=8, decoder_heads=16,
        mask_ratio=args.mask_ratio
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler()

    # ==== 日志 ====
    save_config(vars(args), out_dir / "config.json")
    train_logger = CSVLogger(
        out_dir / "train_log.csv",
        fieldnames=["step", "epoch", "loss_mae", "lr", "mask_ratio", "wall_time"]
    )
    ablog = JSONLAggregator("runs/ablations/ablation_log.jsonl")

    step = 0
    for epoch in trange(args.epochs, desc="Training epochs"):
        loss_avg = train_one_epoch(model, train_loader, optimizer, device,
                                   scaler, use_amp=args.amp)
        step += len(train_loader)

        train_logger.log({
            "step": step,
            "epoch": epoch,
            "loss_mae": float(loss_avg),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "mask_ratio": args.mask_ratio,
            "wall_time": time.time()
        })

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss_avg:.4f}")

        if (epoch+1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scaler, epoch+1, out_dir, args)

    save_checkpoint(model, optimizer, scaler, args.epochs, out_dir, args, final=True)

    train_logger.close()

    ablog.log({
        "exp": exp_name,
        "method": "MAE",
        "mask_ratio": args.mask_ratio,
        "epochs": args.epochs,
        "final_loss": float(loss_avg),
        "encoder_ckpt": str(out_dir / "ckpts/final.pth")
    })
    torch.save(model.encoder.state_dict(), out_dir / "ckpts/final_encoder.pth")
    ablog.close()


if __name__ == "__main__":
    main()