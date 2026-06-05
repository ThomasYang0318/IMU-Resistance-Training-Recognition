from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DSMSTCNConfig:
    input_channels: int = 9
    micro_classes: int = 17
    macro_classes: int = 9
    hidden_channels: int = 64
    num_layers: int = 7
    macro_stages: int = 3
    dropout: float = 0.3


class DilatedResidualLayer(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageTCN(nn.Module):
    def __init__(self, input_channels: int, output_classes: int, hidden_channels: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.in_conv = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.layers = nn.ModuleList(
            DilatedResidualLayer(hidden_channels, dilation=2**idx, dropout=dropout)
            for idx in range(num_layers)
        )
        self.out_conv = nn.Conv1d(hidden_channels, output_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.in_conv(x)
        for layer in self.layers:
            out = layer(out)
        return self.out_conv(out)


class DSMSTCN(nn.Module):
    """Dual-scale multi-stage TCN adapted for 9-axis workout segmentation.

    Input shape is ``(batch, time, channels)``. The model returns one micro
    prediction stage plus multiple macro prediction stages. Macro stage 1 sees
    both IMU features and micro-label probabilities; later stages refine macro
    probabilities, following the MS-TCN pattern.
    """

    def __init__(self, cfg: DSMSTCNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.micro_stage = SingleStageTCN(
            cfg.input_channels,
            cfg.micro_classes,
            cfg.hidden_channels,
            cfg.num_layers,
            cfg.dropout,
        )
        self.macro_stage_1 = SingleStageTCN(
            cfg.input_channels + cfg.micro_classes,
            cfg.macro_classes,
            cfg.hidden_channels,
            cfg.num_layers,
            cfg.dropout,
        )
        self.macro_refine_stages = nn.ModuleList(
            SingleStageTCN(
                cfg.macro_classes,
                cfg.macro_classes,
                cfg.hidden_channels,
                cfg.num_layers,
                cfg.dropout,
            )
            for _ in range(max(0, cfg.macro_stages - 1))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = x.transpose(1, 2)
        micro_logits = self.micro_stage(x)
        micro_probs = F.softmax(micro_logits, dim=1)
        macro_logits = self.macro_stage_1(torch.cat([x, micro_probs], dim=1))
        macro_outputs = [macro_logits]
        for stage in self.macro_refine_stages:
            macro_logits = stage(F.softmax(macro_logits, dim=1))
            macro_outputs.append(macro_logits)
        return micro_logits, macro_outputs


class MSTCN(nn.Module):
    """MS-TCN baseline without micro-label conditioning."""

    def __init__(self, cfg: DSMSTCNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage_1 = SingleStageTCN(
            cfg.input_channels,
            cfg.macro_classes,
            cfg.hidden_channels,
            cfg.num_layers,
            cfg.dropout,
        )
        self.refine_stages = nn.ModuleList(
            SingleStageTCN(
                cfg.macro_classes,
                cfg.macro_classes,
                cfg.hidden_channels,
                cfg.num_layers,
                cfg.dropout,
            )
            for _ in range(max(0, cfg.macro_stages - 1))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
        x = x.transpose(1, 2)
        macro_logits = self.stage_1(x)
        macro_outputs = [macro_logits]
        for stage in self.refine_stages:
            macro_logits = stage(F.softmax(macro_logits, dim=1))
            macro_outputs.append(macro_logits)
        return None, macro_outputs
