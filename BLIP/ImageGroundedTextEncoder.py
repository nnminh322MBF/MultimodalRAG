import torch
import torch.nn as nn
from components import EncoderBlock, CrossAttention


class ImageGroundedTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        max_position_embeddings: int = 77,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        cross_attention_iterval: int = 1,
    ):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.possition_embeddings = nn.Parameter(
            torch.zeros(1, max_position_embeddings, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop_rate)

        stochastic_drop_path = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_dropout=attn_drop_rate,
                    dropout=drop_rate,
                    drop_path=stochastic_drop_path[i],
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.possition_embeddings)
        self.apply(self._init_weights)
        self.cross_idx = set(range(0, depth, cross_attention_iterval))
        self.cross_attn = nn.ModuleList(
            {
                str(i): CrossAttention(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    prj_drop=drop_rate,
                )
                for i in self.cross_idx
            }
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()

        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, token_ids: torch.Tensor, image_feature: torch.Tensor):
        _, seq_len = token_ids.shape
        text_embed = self.word_embeddings(token_ids)
        text_embed = text_embed + self.possition_embeddings[:, :seq_len, :]
        x = self.pos_drop(text_embed)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.cross_idx:
                x = x + self.cross_attn[str(i)](x, image_feature)
        
        x = self.norm(x)

        return x