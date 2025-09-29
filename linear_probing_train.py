import argparse
import json
from pathlib import Path
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from mae_model import PatchEmbed, MAEEncoder


# ------------------------------
# Dataset Loader
# ------------------------------
def get_galaxy10_dataloaders(batch_size, num_workers=4):
    ds = load_dataset("matthieulel/galaxy10_decals")
    train_set, val_set = ds["train"], ds["test"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    ])

    class Galaxy10Dataset(Dataset):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# ------------------------------
# Linear Probe Model
# ------------------------------
class LinearProbe(nn.Module):
    def __init__(self, embed_dim=768, num_classes=10,
                 img_size=224, patch_size=16, in_chans=3,
                 depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.encoder = MAEEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        for p in self.encoder.parameters():
            p.requires_grad = False  # freeze encoder
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, N, D]
        z = self.encoder(x)      # [B, N, D]
        z_mean = z.mean(dim=1)   # global average pooling
        return self.head(z_mean)


# ------------------------------
# Train & Eval
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.05)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataloaders
    train_loader, val_loader = get_galaxy10_dataloaders(batch_size=args.batch_size)

    # Model
    model = LinearProbe(embed_dim=768, num_classes=10).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.encoder.load_state_dict(ckpt, strict=False)
    print("Loaded encoder:", args.ckpt)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    optimizer = optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    outdir = Path("linear_probe_logs")
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "train_log.csv"

    best_acc = 0.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

        for epoch in range(args.epochs):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, criterion)
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)

            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss {tr_loss:.4f}, Acc {tr_acc:.4f} | "
                  f"Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

            writer.writerow([epoch+1, tr_loss, val_loss, tr_acc, val_acc])
            f.flush()

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "linear_probe_best.pth")

    print(f"✅ Finished! Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()