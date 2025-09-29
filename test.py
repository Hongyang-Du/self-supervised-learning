import argparse
import json
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

from mae_model import MAEEncoder  # 只用 encoder


# ------------------------------
# 数据加载
# ------------------------------
def get_galaxy10_dataloaders(batch_size, num_workers=4):
    ds = load_dataset("matthieulel/galaxy10_decals")
    train_set, val_set = ds["train"], ds["test"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    ])

    class Galaxy10Dataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.hf = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.hf)

        def __getitem__(self, idx):
            img = self.hf[idx]["image"]
            label = self.hf[idx]["label"]
            if self.transform:
                img = self.transform(img)
            return img, label

    train_dataset = Galaxy10Dataset(train_set, transform)
    val_dataset = Galaxy10Dataset(val_set, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# ------------------------------
# Linear Probe Model
# ------------------------------
class LinearProbe(nn.Module):
    def __init__(self, encoder, embed_dim=768, num_classes=10):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():  # freeze encoder
            p.requires_grad = False
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # [B,C,H,W] -> patch embedding
        x = self.encoder.patch_embed(x)   # [B,N,D]
        x = self.encoder(x)               # [B,N,D]
        x = x.mean(dim=1)                 # mean pooling
        return self.head(x)


# ------------------------------
# 训练 + 验证
# ------------------------------
def train_one_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(dataloader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(dataloader, desc="Eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to best_encoder.pth")
    parser.add_argument("--config", type=str, default=None, help="path to config.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取 config.json
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 1e-3)
    epochs = config.get("epochs", 100)
    exp_name = config.get("exp_name", "linear_probe")

    outdir = Path("runs/linear_probe") / exp_name
    outdir.mkdir(parents=True, exist_ok=True)

    # 数据
    train_loader, val_loader = get_galaxy10_dataloaders(batch_size=batch_size)

    # encoder
    encoder = MAEEncoder(embed_dim=768, depth=12, num_heads=12).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    encoder.load_state_dict(ckpt, strict=True)

    # probe
    model = LinearProbe(encoder, embed_dim=768, num_classes=10).to(device)

    optimizer = optim.AdamW(model.head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # CSV log
    log_path = outdir / "linear_probe_log.csv"
    with open(log_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        best_acc = 0.0
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion)
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | "
                  f"Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

            writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])
            f_csv.flush()

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), outdir / "linear_probe_best.pth")

        # 保存 final 模型
        torch.save(model.state_dict(), outdir / "linear_probe_final.pth")

    print(f"✅ Finished! Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()