import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset
from mae_model import unpatchify

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

        # === unpatchify ===
    recon = unpatchify(pred, patch_size=16, img_size=224, C=3).detach().cpu()
    mask_expanded = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_dim).cpu()
    mask_img = unpatchify(mask_expanded, patch_size=16, img_size=224, C=3).detach().cpu()  # ✅ 改这里

    # === 反归一化 ===
    def denorm(x):
        return (x * 0.2 + 0.5).clamp(0, 1)

    orig = denorm(x.cpu())[0].permute(1, 2, 0).numpy()
    masked = denorm(x.cpu() * (1 - mask_img))[0].permute(1, 2, 0).numpy()  # ✅ 这下才是真·masked input
    recon = denorm(recon)[0].permute(1, 2, 0).numpy()
    
    # === 保存拼接图 ===
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(orig); plt.title("Original"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(masked); plt.title("Masked"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(recon); plt.title("Reconstruction"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved reconstruction result to {save_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    ckpt_path = "runs/mae_pretrain/debug_run/ckpts/final.pth"  # 改成你的路径
    model = load_model(ckpt_path, device)

    # 从 dataset 里取一张
    ds = load_dataset("matthieulel/galaxy10_decals")
    sample = ds["test"][0]["image"]  # 拿 test split 第一张

    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    save_path = outdir / "sample_recon.png"

    visualize_and_save(model, sample, save_path, device)