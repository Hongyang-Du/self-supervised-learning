import torch
import torch.nn as nn
from mae_model import PatchEmbed, MAEEncoder


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

        # freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # linear classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # image -> patch tokens
        x = self.patch_embed(x)   # [B, N, D]
        # encoder forward
        z = self.encoder(x)       # [B, N, D]
        # global average pooling
        z_mean = z.mean(dim=1)    # [B, D]
        # logits
        return self.head(z_mean)