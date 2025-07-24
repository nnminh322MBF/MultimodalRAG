import torch
import torch.nn as nn
from components import EncoderBlock


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
        text_embed = self.pos_drop(text_embed)




class ImageGroundedTextEncoder(nn.Module):
    """
    Image-Grounded Text Encoder (Option 1):
    Injects cross-attention layers to integrate image features into text features.
    It does NOT contain the ITM head. The ITM head (or any other task-specific projection)
    should be handled by a higher-level module (e.g., a BLIP main class).
    """
    def __init__(self,
                 base_encoder: TextEncoder,
                 cross_interval: int = 1):
        super().__init__()
        # The base_encoder is the unimodal TextEncoder
        self.base = base_encoder
        # Determine at which blocks to inject cross-attention
        self.cross_idx = set(range(0, len(self.base.blocks), cross_interval))

        # Create cross-attention modules, indexed by block number
        self.cross_layers = nn.ModuleDict({
            str(i): CrossAttention(self.base.token_embed.embedding_dim,
                                   self.base.blocks[0].attn.num_heads)
            for i in self.cross_idx
        })
        # Removed self.itm_head as per Option 1 design

    def forward(self, token_ids: torch.Tensor, image_feats: torch.Tensor):
        """
        Forward pass for the ImageGroundedTextEncoder.
        Args:
            token_ids (torch.Tensor): Input text token IDs (B, T).
            image_feats (torch.Tensor): Image features (B, N_img, dim).
        Returns:
            torch.Tensor: The image-grounded text features (B, T, dim).
                          The ITM logit is NOT returned here.
        """
        # Initial text embedding from the base TextEncoder's embedding layer and positional embedding
        x = self.base.token_embed(token_ids) + self.base.pos_embed[:, :token_ids.size(1)]
        x = self.base.drop(x) # Apply dropout

        # Iterate through the base TextEncoder's blocks
        for i, blk in enumerate(self.base.blocks):
            # Apply the standard text block (self-attention and MLP)
            x = blk(x)
            # If the current block index is in cross_idx, inject cross-attention
            if i in self.cross_idx:
                # Add the output of cross-attention to the current text features
                # This allows image features to ground the text representation
                x = x + self.cross_layers[str(i)](x, image_feats)

        # Apply final layer normalization from the base TextEncoder
        x = self.base.norm(x)

        # Return only the image-grounded text features.
        # The ITM head and logit calculation are removed as they belong to a higher-level module.
        return x  # (B, T, dim)
