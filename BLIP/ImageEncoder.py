import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from components import Block


class ImageEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        assert (
            image_size % patch_size == 0
        ), "Image dimentions must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            3, embed_dim, stride=patch_size, kernel_size=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        drp = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_dropout=attn_drop_rate,
                    dropout=drop_rate,
                    drop_path=drp[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W)
        returns: (B, num_patches+1, embed_dim)  — đầu tiên là CLS token
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, embed_dim, H/ph, W/pw]
        x = x.flatten(2).transpose(1, 2)  # [B, C, embed_dim]
        self.cls_token = self.cls_token.expand(B, -1, -1)

        x = torch.concatenate([self.cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x
