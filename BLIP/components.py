import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor):
        if self.drop_prob == 0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            size=shape, dtype=x.dtype, device=x.device
        )
        random_tensor = torch.floor_(random_tensor)

        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop_out=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        x = self.drop_out(x)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads, qkv_bias=False, attn_dropout=0.0, prj_dropout=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.prj = nn.Linear(dim, dim)
        self.prj_drop = nn.Dropout(prj_dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.tensor = None):
        B, N, C = x.shape
        qkv: torch.Tensor = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn: torch.Tensor = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (
            (attn @ v).transpose(1, 2).reshape(B, N, C)
        )  # [Batch_size, N, num_heads, d_k]

        x = self.prj(x)
        x = self.prj_drop(x)

        return x


class CausalAttention(SelfAttention):
    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        causual_mask = torch.tril(torch.ones(N, N, dtype=x.dtype, device=x.device))
        return super().forward(x, causual_mask)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, prj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.prj = nn.Linear(dim, dim)
        self.prj_drop = nn.Dropout(prj_drop)

    def forward(
        self, query: torch.Tensor, context: torch.Tensor, cross_attn_mask=None
    ):
        """
        input:
        - context: image_feature
        - query: text_feature
        """
        _, Ni, _ = context.shape
        B, Nt, C = query.shape

        q: torch.Tensor = self.q(query)  # [Batch_size, Nt, dim]
        q = q.reshape(B, Nt, 1, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )  # [1, Batch_size, num_heads, N_t, d_k]

        kv: torch.Tensor = self.kv(context)
        kv = kv.reshape(B, Ni, 2, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )  # [2, Batch_size, num_heads, N_i, d_k]
        k, v = kv[0], kv[1]

        attn: torch.Tensor = (q @ k.transpose(-2, -1)) * self.scale

        if cross_attn_mask is not None:
            attn = attn.masked_fill(cross_attn_mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)  # [Batch_size, num_heads, N_t, N_i]
        attn = self.attn_drop(attn)
        x = attn @ v  # [Batch_size, num_heads, N_t, d_k]
        x = x.transpose(1, 2).reshape(B, Nt, C)
        x = self.prj(x)
        x = self.prj_drop(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        attn_dropout=0.0,
        dropout=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            prj_dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = (
            DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        )
        hidden_dim = int(mlp_ratio * dim)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_dim, drop_out=dropout)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        attn_dropout=0.0,
        dropout=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.causual_attn = CausalAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            prj_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_dropout,
            prj_drop=dropout,
        )
        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            drop_out=dropout,
        )

    def forward(self, x: torch.Tensor, encoder_hidden_state: torch.Tensor):
        x = x + self.drop_path(self.causual_attn(self.norm1(x)))
        x = x + self.drop_path(self.cross_attn(self.norm2(x), encoder_hidden_state))
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x