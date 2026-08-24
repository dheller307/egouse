"""Causal GRU student: shared slot encoder + per-hand multi-class head."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lib.student_sequence_data import SLOT_DIM_BASE, SLOT_DIM_ONEHOT


class StudentHandGRU(nn.Module):
  def __init__(
    self,
    *,
    vocab_size: int,
    slot_dim: int = SLOT_DIM_ONEHOT,
    d_slot: int = 16,
    hidden: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    num_classes: int = 34,
    hand_cond: str = "onehot",
    hand_embed_dim: int = 8,
    label_mode: str = "exclusive",
  ) -> None:
    super().__init__()
    self.vocab_size = vocab_size
    self.hand_cond = hand_cond
    self.label_mode = label_mode
    self.hand_embed_dim = hand_embed_dim
    self.slot_encoder = nn.Sequential(
      nn.Linear(slot_dim, d_slot),
      nn.ReLU(),
    )
    gru_in = vocab_size * d_slot
    self.hand_embed: nn.Embedding | None = None
    if hand_cond == "embed":
      self.hand_embed = nn.Embedding(2, hand_embed_dim)
      gru_in += hand_embed_dim
    self.gru = nn.GRU(
      gru_in,
      hidden,
      num_layers=num_layers,
      batch_first=True,
      dropout=dropout if num_layers > 1 else 0.0,
    )
    self.head = nn.Linear(hidden, num_classes)

  def forward(
    self,
    x: torch.Tensor,
    lengths: torch.Tensor,
    hand_id: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """
    x: (B, T, V, slot_dim)
    lengths: (B,) int64
    hand_id: (B,) int64 in {0,1} when hand_cond == embed
    returns logits (B, T, num_classes)
    """
    b, t, v, _d = x.shape
    enc = self.slot_encoder(x)
    enc = enc.reshape(b, t, v * enc.shape[-1])
    if self.hand_embed is not None:
      if hand_id is None:
        raise ValueError("hand_id required when hand_cond=embed")
      h = self.hand_embed(hand_id)
      h = h.unsqueeze(1).expand(-1, t, -1)
      enc = torch.cat([enc, h], dim=-1)
    lengths_cpu = lengths.detach().to("cpu")
    packed = pack_padded_sequence(
      enc, lengths_cpu, batch_first=True, enforce_sorted=False
    )
    out_packed, _ = self.gru(packed)
    out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=t)
    return self.head(out)
