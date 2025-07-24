import torch
import torch.nn as nn
from components import EncoderBlock


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 512,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.word_embeddings = nn.Parameter(vocab_size, embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.drop = nn.Dropout(drop_rate)

        stochastic_drop_path = torch.linspace(0, drop_path_rate, depth)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=attn_drop_rate,
                    dropout=drop_rate,
                    drop_path=stochastic_drop_path[i],
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.max_len = max_len

        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, token_ids: torch.Tensor):
        _, seq_len = token_ids.shape
        pos = self.position_embeddings[:, :seq_len, :]
        word_embed = self.word_embeddings(token_ids) + pos
        x = self.drop(word_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
