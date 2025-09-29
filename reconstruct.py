import torch
from torchvision import transforms
from pathlib import Path
from datasets import load_dataset

from mae_model import MAE, unpatchify  # 假设你有这两个

def load_model(ckpt_path, device):
    model = MAE(
        img_size=224, patch_size=16, in_chans=3,
        embed_dim=768, depth=12, num_heads=12,
        decoder_dim=512, decoder_depth=8, decoder_heads=16,
        mask_ratio=0.75
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def visualize_and_save(model, img, save_path, device="cuda"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.2, 0.2, 0.2])
    ])
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        loss, pred, mask = model(x)

    recon = unpatchify(pred, patch_size=16, img_size=224, C=3).cpu()
    mask_expanded = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_dim).cpu()
    mask_img = unpatchify(mask_expanded, patch_size=16, img_size=224, C=3).cpu()

    def denorm(x):
        return (x * 0.2 + 0.5).clamp(0, 1)

    orig = denorm(x.cpu())[0].permute(1, 2, 0).numpy()
    rec_np = denorm(recon)[0].permute(1, 2, 0).numpy()
    mask_np = mask_img[0, 0].numpy()  # 0 或 1

    masked_np = orig.copy()
    masked_np[mask_np == 1] = 0

    merged = rec_np.copy()
    merged[mask_np == 0] = orig[mask_np == 0]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1); plt.imshow(orig); plt.title("Original"); plt.axis("off")
    plt.subplot(1, 4, 2); plt.imshow(masked_np); plt.title("Masked"); plt.axis("off")
    plt.subplot(1, 4, 3); plt.imshow(rec_np); plt.title("Reconstruction"); plt.axis("off")
    plt.subplot(1, 4, 4); plt.imshow(merged); plt.title("Reconstruction+Visible"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved reconstruction to {save_path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_dirs = [
        "runs/mae_pretrain/mask_0.25",
        "runs/mae_pretrain/mask_0.5",
        "runs/mae_pretrain/mask_0.75",
    ]

    ds = load_dataset("matthieulel/galaxy10_decals")
    sample = ds["test"][80]["image"]

    outdir = Path("reconstruct_results")
    outdir.mkdir(parents=True, exist_ok=True)

    for exp in exp_dirs:
        ckpt_path = Path(exp) / "ckpts" / "final.pth"
        if not ckpt_path.exists():
            print(f"⚠️ {ckpt_path} 不存在，跳过")
            continue

        print(f"Running recon for {exp}")
        model = load_model(str(ckpt_path), device)
        save_path = outdir / f"recon_{Path(exp).name}.png"
        visualize_and_save(model, sample, save_path, device)

if __name__ == "__main__":
    main()