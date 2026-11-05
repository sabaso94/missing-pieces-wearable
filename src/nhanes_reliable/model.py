"""Fusion survival network: a day encoder for the wearable, an MLP for the tabular blocks, one survival head.

Each valid day is encoded on its own by a shared 1D CNN, and the day embeddings are averaged over the valid
days only. So the model takes any number of days (1-7), and the order of days does not matter.
"""

import math

import torch
from torch import nn

from .survival import N_INTERVALS

N_FLAGS = 4  # block-present flags: wearable, labs, exam, questionnaire


class DayEncoder(nn.Module):
    """1D CNN over one day of minutes: (n, 2, 1440) -> (n, emb_dim).

    Two fixed clock channels (sine and cosine of the time of day) are added, so the network can tell night
    from day after the convolutions; global average pooling alone would lose that.
    """

    def __init__(self, emb_dim: int = 64):
        super().__init__()
        minutes = torch.arange(1440) / 1440 * 2 * math.pi
        self.register_buffer("clock", torch.stack([torch.sin(minutes), torch.cos(minutes)]))
        layers, channels = [], [4, 16, 32, 64, 64]
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            # Each block halves the time axis: 1440 -> 720 -> 360 -> 180 -> 90 steps.
            layers += [nn.Conv1d(c_in, c_out, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(c_out), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(channels[-1], emb_dim)

    def forward(self, day: torch.Tensor) -> torch.Tensor:
        x = torch.cat([day, self.clock.expand(len(day), -1, -1)], dim=1)
        return self.out(self.conv(x).mean(dim=2))


class FusionSurvivalNet(nn.Module):
    """Per-interval hazard logits from the wearable days, the tabular features and the block-present flags."""

    def __init__(self, n_tab: int, use_wearable: bool, n_intervals: int = N_INTERVALS, emb_dim: int = 64):
        super().__init__()
        self.use_wearable = use_wearable
        self.emb_dim = emb_dim
        self.days = DayEncoder(emb_dim) if use_wearable else None
        self.tab = nn.Sequential(nn.Linear(n_tab, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 64), nn.ReLU())
        head_in = 64 + N_FLAGS + (emb_dim if use_wearable else 0)
        self.head = nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, n_intervals))

    def forward(self, days: torch.Tensor | None, day_mask: torch.Tensor | None, tab: torch.Tensor,
                flags: torch.Tensor) -> torch.Tensor:
        """days (B, 7, 2, 1440), day_mask (B, 7) bool, tab (B, n_tab), flags (B, 4) -> logits (B, n_intervals)."""
        parts = [self.tab(tab)]
        if self.use_wearable:
            # The wearable counts as present only if its flag is on and at least one day is valid.
            present = flags[:, 0] * day_mask.any(dim=1).float()
            flags = torch.cat([present[:, None], flags[:, 1:]], dim=1)
            mask = day_mask & (present[:, None] > 0)
            embedding = torch.zeros(len(tab), self.emb_dim, device=tab.device)
            if mask.any():
                # Encode only the valid days, then average them per person.
                encoded = self.days(days[mask])
                people = mask.nonzero()[:, 0]
                embedding = embedding.index_add(0, people, encoded)
                embedding = embedding / mask.sum(dim=1).clamp(min=1)[:, None].float()
            parts.append(embedding)
        parts.append(flags)
        return self.head(torch.cat(parts, dim=1))
