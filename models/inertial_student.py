from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_channels: int = 6
    num_classes: int = 8
    patch_size: int = 10
    embed_dim: int = 128
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: int = 2
    dropout: float = 0.1


class PatchEmbed1D(nn.Module):
    def __init__(self, input_channels: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * input_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        usable = (t // self.patch_size) * self.patch_size
        x = x[:, :usable, :]
        n = usable // self.patch_size
        x = x.reshape(b, n, self.patch_size * c)
        return self.proj(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_in = self.norm1(x)
        attn_out, attn_weights = self.attn(attn_in, attn_in, attn_in, need_weights=True, average_attn_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class InertialStudent(nn.Module):
    def __init__(self, cfg: ModelConfig, window_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed1D(cfg.input_channels, cfg.patch_size, cfg.embed_dim)

        num_patches = window_size // cfg.patch_size
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=cfg.embed_dim,
                    heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        x = self.patch_embed(x)
        b = x.shape[0]
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.drop(x + self.pos[:, : x.shape[1], :])

        attn_maps: List[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x)
            attn_maps.append(attn)

        x = self.norm(x)
        logits = self.head(x[:, 0])
        if return_attention:
            return logits, attn_maps
        return logits


def export_student_to_onnx(
    model: nn.Module,
    output_path: str,
    window_size: int,
    input_channels: int,
    opset_version: int = 17,
) -> None:
    model.eval()
    dummy = torch.randn(1, window_size, input_channels)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["imu_window"],
        output_names=["logits"],
        dynamic_axes={"imu_window": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset_version,
    )
