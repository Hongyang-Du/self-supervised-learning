import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
import pandas as pd
from datasets import load_dataset

from mae_model import MAE, unpatchify, patchify  # 假设你有这两个

def load_model_from_ckpt(ckpt_path, device, model_args):
    model = MAE(**model_args).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def compute_psnr(recon, target):
    mse = F.mse_loss(recon, target, reduction="mean")
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()

def evaluate_model(model, x, val_loader, device):
    # train_loss on x
    with torch.no_grad():
        loss_train, pred_train, mask_train = model(x)
    train_loss = loss_train.item()

    # reconstruction
    recon = unpatchify(pred_train, patch_size=model.patch_size, img_size=x.shape[2], C=x.shape[1]).to(device)

    # psnr
    psnr = compute_psnr(recon, x)

    # val_loss on validation set
    losses = []
    for imgs in val_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            loss_val, _, _ = model(imgs)
        losses.append(loss_val.item())
    val_loss = sum(losses) / len(losses)

    return train_loss, val_loss, psnr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = {
        "img_size": 224,
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_heads": 16,
        "mask_ratio": 0.75,
    }

    exp_dirs = [
        "runs/mae_pretrain/mask_0.25",
        "runs/mae_pretrain/mask_0.5",
        "runs/mae_pretrain/mask_0.75",
    ]

    # load validation dataset
    ds = load_dataset("matthieulel/galaxy10_decals")
    val_set = ds["test"]
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2])
    ])
    class ValDataset(torch.utils.data.Dataset):
        def __init__(self, hf, transform):
            self.hf = hf
            self.transform = transform
        def __len__(self):
            return len(self.hf)
        def __getitem__(self, idx):
            img = self.hf[idx]["image"]
            return self.transform(img)
    val_loader = torch.utils.data.DataLoader(ValDataset(val_set, transform), batch_size=32, shuffle=False)

    # pick a sample for train_loss / psnr
    train_ds = ds["train"]
    sample = train_ds[0]["image"]
    x = transform(sample).unsqueeze(0).to(device)

    all_records = []
    for exp in exp_dirs:
        ckpt_folder = Path(exp) / "ckpts"
        if not ckpt_folder.exists():
            print("⚠️ skip no folder", ckpt_folder)
            continue

        ckpt_list = sorted(ckpt_folder.glob("checkpoint_epoch_*.pth"))
        # optionally also include final.pth if exists
        final_p = ckpt_folder / "final.pth"
        if final_p.exists():
            ckpt_list.append(final_p)

        for ckpt_path in ckpt_list:
            print("→ Evaluating:", exp, ckpt_path.name)
            model = load_model_from_ckpt(str(ckpt_path), device, model_args)
            train_loss, val_loss, psnr = evaluate_model(model, x, val_loader, device)
            all_records.append({
                "exp": Path(exp).name,
                "ckpt": ckpt_path.name,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "psnr": psnr
            })

    # 保存结果
    df = pd.DataFrame(all_records)
    out = Path("recon_from_all_ckpts") / "metrics_all_epochs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Saved all metrics to", out)

if __name__ == "__main__":
    main()