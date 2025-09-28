import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from mae_model import MAE   # 确保和训练时定义一致


def load_model(ckpt_path, device):
    # 初始化 MAE（和训练时参数要一致！）
    model = MAE(
        img_size=224, patch_size=16, in_chans=3,
        embed_dim=768, depth=12, num_heads=12,
        decoder_dim=512, decoder_depth=8, decoder_heads=16,
        mask_ratio=0.75  # 这里值无所谓，forward 会重新采样 mask
    ).to(device)

    # 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def visualize_reconstruction(model, img_path, device="cuda"):
    # transform 与训练保持一致
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2])
    ])

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        loss, pred, mask = model(x)

    # ---- unpatchify 恢复图片 ----
    recon = model.unpatchify(pred).detach().cpu()
    mask = mask.unsqueeze(-1).repeat(1,1,model.patch_embed.patch_dim).cpu()
    mask = model.unpatchify(mask).detach()

    # 还原归一化
    def denorm(x):
        return (x * 0.2 + 0.5).clamp(0,1)

    # 原图
    orig = denorm(x.cpu())[0].permute(1,2,0).numpy()
    # Masked 输入
    im_masked = denorm(x.cpu()*(1-mask))[0].permute(1,2,0).numpy()
    # 重建结果
    recon = denorm(recon)[0].permute(1,2,0).numpy()

    # ---- 可视化 ----
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(orig); plt.title("Original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(im_masked); plt.title("Masked"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(recon); plt.title("Reconstruction"); plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model = load_model(args.ckpt, args.device)
    visualize_reconstruction(model, args.img, args.device)