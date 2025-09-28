import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from mae_model import MAE   # 确保和训练时一致


def load_model(ckpt_path, device):
    model = MAE(
        img_size=224, patch_size=16, in_chans=3,
        embed_dim=768, depth=12, num_heads=12,
        decoder_dim=512, decoder_depth=8, decoder_heads=16,
        mask_ratio=0.75
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def visualize_and_save(model, img_path, save_path, device="cuda"):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2])
    ])

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        loss, pred, mask = model(x)

    # ---- unpatchify ----
    recon = model.unpatchify(pred).detach().cpu()
    mask = mask.unsqueeze(-1).repeat(1,1,model.patch_embed.patch_dim).cpu()
    mask = model.unpatchify(mask).detach()

    # 反归一化
    def denorm(x):
        return (x * 0.2 + 0.5).clamp(0,1)

    orig = denorm(x.cpu())[0].permute(1,2,0).numpy()
    im_masked = denorm(x.cpu()*(1-mask))[0].permute(1,2,0).numpy()
    recon = denorm(recon)[0].permute(1,2,0).numpy()

    # 保存拼接图
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(orig); plt.title("Original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(im_masked); plt.title("Masked"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(recon); plt.title("Reconstruction"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved reconstruction result to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--outdir", type=str, default="results", help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model = load_model(args.ckpt, args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    save_path = outdir / (Path(args.img).stem + "_recon.png")
    visualize_and_save(model, args.img, save_path, args.device)