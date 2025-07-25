import torch
import torch.nn as nn
from components import DecoderBlock


class ImageGroundedTextDecoder(nn.Module):
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
        cross_attention_interval: int = 1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, max_position_embeddings, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop_rate)
        stochastic_drop_path = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
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
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor, image_feature: torch.Tensor):
        _, seq_len = token_ids.shape
        pos_embed = self.position_embeddings[:, :seq_len, :]
        text_embed = self.word_embeddings(token_ids)
        text_embed = text_embed + pos_embed
        x = self.pos_drop(text_embed)

        for blk in self.blocks:
            x = blk(x, image_feature)

        x = self.norm(x)
        logits = self.head(x)

        return logits
