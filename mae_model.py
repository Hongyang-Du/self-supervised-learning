import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

# ------------------------
# 1) Patchify / Unpatchify
# ------------------------
class PatchEmbed(nn.Module):
    """Conv2d实现的线性投影，等价于 unfold+Linear；P=16"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_dim = patch_size * patch_size * in_chans 
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, N, D]
        x = self.proj(x)                  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x

def patchify(imgs, patch_size=16):
    # imgs: [B, C, 224, 224] -> [B, N, P*P*C]
    B, C, H, W = imgs.shape
    assert H == W and H % patch_size == 0
    p = patch_size
    h = w = H // p
    x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1)
    # [B, h, w, p, p, C] -> [B, h*w, p*p*C]
    x = x.reshape(B, h * w, p * p * C)
    return x

def unpatchify(x, patch_size=16, img_size=224, C=3):
    # x: [B, N, p*p*C] -> [B, C, 224, 224]
    B, N, PPc = x.shape
    p = patch_size
    h = w = img_size // p
    assert N == h * w
    x = x.reshape(B, h, w, p, p, C).permute(0, 5, 1, 3, 2, 4)
    x = x.reshape(B, C, h * p, w * p)
    return x

# ------------------------
# 2) 随机遮挡（按论文默认 0.75）
# ------------------------
def random_masking(x, mask_ratio=0.75):
    """
    x: [B, N, D] tokens
    返回：x_vis（可见tokens）、mask（1=mask,0=visible）、ids_restore（把[可见+mask]还原到原顺序的索引）
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)  # 每张图独立打乱
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_vis = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)  # 按原顺序

    return x_vis, mask, ids_restore

# ------------------------
# 3) MAE 编码器（ViT-B/16，仅可见tokens）
# ------------------------
class MAEEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.pos_embed_vis = nn.Parameter(torch.zeros(1, 14*14, embed_dim))  # 224/16=14
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed_vis, std=0.02)

    def forward(self, x_vis):
        # x_vis: [B, N_vis, D]
        x = x_vis + self.pos_embed_vis[:, :x_vis.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # [B, N_vis, D]

# ------------------------
# 4) MAE 解码器（轻量ViT，接收[可见latent + mask token]）
# ------------------------
class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=768, decoder_dim=512, depth=8, num_heads=16, patch_size=16, in_chans=3):
        super().__init__()
        self.proj = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed_all = nn.Parameter(torch.zeros(1, 14*14, decoder_dim))
        self.blocks = nn.ModuleList([
            Block(decoder_dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_size * patch_size * in_chans, bias=True)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_all, std=0.02)

    def forward(self, z_vis, ids_restore):
        """
        z_vis: [B, N_vis, Denc]
        ids_restore: [B, N]，将 [可见+mask] 还原为原顺序的索引
        """
        B, N = ids_restore.shape
        z = self.proj(z_vis)

        # 准备N个位置，fill到原顺序
        expand_mask = self.mask_token.repeat(B, N - z.size(1), 1)
        x_ = torch.cat([z, expand_mask], dim=1)  # [B, N_vis+N_mask, Ddec]

        # 复原顺序
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.size(-1)))
        x = x + self.pos_embed_all

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pred = self.head(x)  # [B, N, p*p*C]
        return pred

# ------------------------
# 5) 整体 MAE（训练时只对被遮挡补丁做MSE）
# ------------------------
class MAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_dim=512, decoder_depth=8, decoder_heads=16, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.encoder = MAEEncoder(embed_dim, depth, num_heads)
        self.decoder = MAEDecoder(embed_dim, decoder_dim, decoder_depth, decoder_heads, patch_size, in_chans)

    def forward(self, imgs):
        # 1) tokens
        x = self.patch_embed(imgs)                 # [B, N, D]
        # 2) 随机遮挡
        x_vis, mask, ids_restore = random_masking(x, self.mask_ratio)
        # 3) 编码（仅可见）
        z_vis = self.encoder(x_vis)                # [B, N_vis, D]
        # 4) 解码（重建所有位置）
        pred = self.decoder(z_vis, ids_restore)    # [B, N, p*p*C]
        # 5) 计算像素回归目标（默认对原始像素patch回归；可替换为log/arcsinh后的像素）
        target = patchify(imgs, self.patch_size)   # [B, N, p*p*C]
        loss = (F.mse_loss(pred, target, reduction='none').mean(dim=-1) * mask).sum() / mask.sum()
        return loss, pred, mask