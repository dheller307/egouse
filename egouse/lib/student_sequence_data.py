"""Pivot long features.csv into per-hand (T, num_objects, 15) tensors for GRU student."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lib.paths import EGOUSE_ROOT
from lib.student_features import FEATURE_COLUMNS

SENTINEL = -1.0
NUM_FEATURE_COLS = len(FEATURE_COLUMNS)
OBJ_VALID_IDX = NUM_FEATURE_COLS  # 14
NUM_OBJ_CHANNELS = NUM_FEATURE_COLS + 1  # features + obj_valid
HAND_ONEHOT_DIM = 2
SLOT_DIM_BASE = NUM_OBJ_CHANNELS  # 14 feats + obj_valid
SLOT_DIM_ONEHOT = NUM_OBJ_CHANNELS + HAND_ONEHOT_DIM
# Back-compat alias (one-hot layout)
SLOT_DIM = SLOT_DIM_ONEHOT

HAND_COND_CHOICES = ("onehot", "embed", "none")
STREAM_TARGET_CHOICES = ("per_hand", "merged")


def hand_side_to_id(hand_side: str) -> int:
  if hand_side == "left":
    return 0
  if hand_side == "right":
    return 1
  raise ValueError(f"unknown hand_side: {hand_side!r}")


def slot_dim_for_hand_cond(hand_cond: str) -> int:
  if hand_cond == "onehot":
    return SLOT_DIM_ONEHOT
  if hand_cond in ("embed", "none"):
    return SLOT_DIM_BASE
  raise ValueError(f"unknown hand_cond: {hand_cond!r}")

# Features that are undefined when object det / pair geometry is missing.
PAIR_GEOMETRY_COLUMNS = frozenset(
  {
    "hand_obj_iou",
    "hand_obj_center_dist_norm",
    "area_ratio",
    "obj_score",
    "rel_speed",
    "iou_roll_mean",
    "dist_roll_mean",
    "dist_roll_std",
    "dist_roll_slope",
  }
)

DEFAULT_PROMPTS_PATH = EGOUSE_ROOT / "config" / "hot3d_yoloe_prompts.json"


def load_object_vocab(prompts_path: Path | None = None) -> tuple[list[str], dict[str, int]]:
  path = prompts_path or DEFAULT_PROMPTS_PATH
  data = json.loads(path.read_text())
  names = sorted(p["instance_name"] for p in data["prompts"])
  obj_to_id = {name: i + 1 for i, name in enumerate(names)}  # 0 = no contact
  return names, obj_to_id


def save_object_vocab(out_path: Path, vocab: list[str]) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps({"object_vocab": vocab}, indent=2))


def load_seq_list(path: Path) -> list[str]:
  return [
    line.strip()
    for line in path.read_text().splitlines()
    if line.strip() and not line.startswith("#")
  ]


def _sanitize_row_features(row: pd.Series) -> np.ndarray:
  """Return (14,) float features with sentinel fill for missing pair geometry."""
  out = np.empty(NUM_FEATURE_COLS, dtype=np.float32)
  for i, col in enumerate(FEATURE_COLUMNS):
    v = row.get(col)
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
      out[i] = SENTINEL if col in PAIR_GEOMETRY_COLUMNS else SENTINEL
    else:
      fv = float(v)
      if col in PAIR_GEOMETRY_COLUMNS and not np.isfinite(fv):
        out[i] = SENTINEL
      else:
        out[i] = fv
  return out


def _hand_onehot(hand_side: str) -> np.ndarray:
  if hand_side == "left":
    return np.array([1.0, 0.0], dtype=np.float32)
  if hand_side == "right":
    return np.array([0.0, 1.0], dtype=np.float32)
  raise ValueError(f"unknown hand_side: {hand_side!r}")


# Feature column indices for augmentation (exclude hand_contact=4, obj_present_frac=13)
_AUG_JITTER_FEATURE_IDX = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
_AUG_DROPOUT_FEATURE_IDX = list(range(NUM_FEATURE_COLS))


def augment_hand_stream_features(
  X: np.ndarray,
  *,
  feature_jitter: float,
  feature_channel_dropout: float,
  rng: np.random.Generator,
) -> np.ndarray:
  """Apply jitter + channel dropout on (T, V, slot_dim) hand stream tensor."""
  if feature_jitter <= 0 and feature_channel_dropout <= 0:
    return X
  out = X.copy()
  t_len, vocab_size, slot_dim = out.shape
  feat_end = min(NUM_FEATURE_COLS, slot_dim)
  for t in range(t_len):
    for v in range(vocab_size):
      if out[t, v, OBJ_VALID_IDX] <= 0:
        continue
      slot = out[t, v, :feat_end]
      if feature_jitter > 0:
        for fi in _AUG_JITTER_FEATURE_IDX:
          if fi >= feat_end:
            continue
          if slot[fi] != SENTINEL and np.isfinite(slot[fi]):
            slot[fi] += rng.normal(0.0, feature_jitter)
      if feature_channel_dropout > 0:
        for fi in _AUG_DROPOUT_FEATURE_IDX:
          if fi >= feat_end:
            continue
          if rng.random() < feature_channel_dropout:
            slot[fi] = SENTINEL
      out[t, v, :feat_end] = slot
  return out


def hand_stream_from_features_csv(
  feat_path: Path,
  hand_side: str,
  object_vocab: list[str],
  object_to_id: dict[str, int],
  *,
  hand_cond: str = "onehot",
  allow_unknown_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
  """
  Build X (T, V, slot_dim), y (T,), t_sec (T,), timestamp_ns list for one hand.

  hand_cond onehot: 17-D slots (14 + obj_valid + left/right one-hot).
  hand_cond embed|none: 15-D slots (hand identity via model embedding or absent).
  """
  if hand_cond not in HAND_COND_CHOICES:
    raise ValueError(hand_cond)
  slot_dim = slot_dim_for_hand_cond(hand_cond)
  df = pd.read_csv(feat_path)
  hand_df = df[df["hand_side"] == hand_side].copy()
  if hand_df.empty:
    raise ValueError(f"No rows for hand_side={hand_side} in {feat_path}")

  vocab_size = len(object_vocab)
  name_to_slot = {name: j for j, name in enumerate(object_vocab)}

  hand_df["slot"] = hand_df["object_name"].map(name_to_slot)
  hand_df = hand_df[hand_df["slot"].notna()].copy()
  hand_df["slot"] = hand_df["slot"].astype(int)

  timestamps = np.sort(hand_df["timestamp_ns"].unique())
  T = len(timestamps)
  ts_to_idx = {int(ts): i for i, ts in enumerate(timestamps)}

  X = np.full((T, vocab_size, slot_dim), SENTINEL, dtype=np.float32)
  X[:, :, OBJ_VALID_IDX] = 0.0
  y = np.zeros(T, dtype=np.int64)
  t_sec = np.zeros(T, dtype=np.float64)

  feat_mat = hand_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32).copy()
  bad = ~np.isfinite(feat_mat)
  feat_mat[bad] = SENTINEL

  slots = hand_df["slot"].to_numpy(dtype=int)
  ts_idx = hand_df["timestamp_ns"].map(lambda t: ts_to_idx[int(t)]).to_numpy(dtype=int)

  for row_i in range(len(hand_df)):
    ti = int(ts_idx[row_i])
    si = int(slots[row_i])
    X[ti, si, :NUM_FEATURE_COLS] = feat_mat[row_i]
    X[ti, si, OBJ_VALID_IDX] = 1.0

  if hand_cond == "onehot":
    hand_oh = _hand_onehot(hand_side)
    X[:, :, NUM_OBJ_CHANNELS : NUM_OBJ_CHANNELS + HAND_ONEHOT_DIM] = hand_oh

  if hand_df["y_interacting"].isna().any() and not allow_unknown_labels:
    raise ValueError("Unannotated labels cannot be used as training negatives")
  pos = hand_df[hand_df["y_interacting"].fillna(0).astype(int) == 1]
  multi = pos.groupby("timestamp_ns").size()
  if (multi > 1).any():
    raise ValueError(f"{feat_path} hand={hand_side}: timesteps with >1 positive")

  for ts, g in pos.groupby("timestamp_ns", sort=False):
    ti = ts_to_idx[int(ts)]
    oname = str(g.iloc[0]["object_name"])
    y[ti] = object_to_id.get(oname, 0)

  t_sec_df = hand_df.groupby("timestamp_ns", sort=False)["t_sec"].first()
  for ts, val in t_sec_df.items():
    t_sec[ts_to_idx[int(ts)]] = float(val)

  return X, y, t_sec, [int(t) for t in timestamps]


def merged_stream_from_features_csv(
  feat_path: Path,
  object_vocab: list[str],
  object_to_id: dict[str, int],
  *, allow_unknown_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
  """
  One stream per sequence: pool L/R features per (t, object); multi-label y (T, V).

  Features: per slot, keep the hand row with highest finite hand_obj_iou (else any row).
  Labels: y[t, j]=1 per object if any hand has teacher ON at t (multiple j allowed).
  """
  slot_dim = SLOT_DIM_BASE
  df = pd.read_csv(feat_path)
  if df.empty:
    raise ValueError(f"empty features {feat_path}")

  if df["y_interacting"].isna().any() and not allow_unknown_labels:
    raise ValueError("Unknown labels cannot be used for training")

  vocab_size = len(object_vocab)
  name_to_slot = {name: j for j, name in enumerate(object_vocab)}

  df = df.copy()
  df["slot"] = df["object_name"].map(name_to_slot)
  df = df[df["slot"].notna()].copy()
  df["slot"] = df["slot"].astype(int)

  timestamps = np.sort(df["timestamp_ns"].unique())
  T = len(timestamps)
  ts_to_idx = {int(ts): i for i, ts in enumerate(timestamps)}
  df["_ti"] = df["timestamp_ns"].map(lambda t: ts_to_idx[int(t)]).astype(int)

  X = np.full((T, vocab_size, slot_dim), SENTINEL, dtype=np.float32)
  X[:, :, OBJ_VALID_IDX] = 0.0
  y_multi = np.zeros((T, vocab_size), dtype=np.float32)
  t_sec = np.zeros(T, dtype=np.float64)

  feat_mat = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32).copy()
  bad = ~np.isfinite(feat_mat)
  feat_mat[bad] = SENTINEL
  row_pos = {idx: i for i, idx in enumerate(df.index)}

  for (ti, si), sg in df.groupby(["_ti", "slot"], sort=False):
    ti = int(ti)
    si = int(si)
    t_sec[ti] = float(sg["t_sec"].iloc[0])
    sub_iou = sg["hand_obj_iou"].to_numpy(dtype=float).copy()
    sub_iou[~np.isfinite(sub_iou)] = -1.0
    pick_idx = int(sg.index[int(np.argmax(sub_iou))])
    ri = row_pos[pick_idx]
    X[ti, si, :NUM_FEATURE_COLS] = feat_mat[ri]
    X[ti, si, OBJ_VALID_IDX] = 1.0
    if (sg["y_interacting"].fillna(0).astype(int) == 1).any():
      y_multi[ti, si] = 1.0

  return X, y_multi, t_sec, [int(t) for t in timestamps]


class MergedSeqDataset(Dataset):
  """One item per sequence (merged L/R targets)."""

  def __init__(
    self,
    seq_ids: list[str],
    features_root: Path,
    object_vocab: list[str] | None = None,
    object_to_id: dict[str, int] | None = None,
  ) -> None:
    if object_vocab is None or object_to_id is None:
      object_vocab, object_to_id = load_object_vocab()
    self.features_root = features_root
    self.object_vocab = object_vocab
    self.object_to_id = object_to_id
    self.seq_ids = list(seq_ids)
    self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]] = {}
    print(f"MergedSeqDataset: preloading {len(self.seq_ids)} sequences...", flush=True)
    for i, seq_id in enumerate(self.seq_ids):
      feat_path = self.features_root / seq_id / "features.csv"
      self._cache[seq_id] = merged_stream_from_features_csv(
        feat_path, self.object_vocab, self.object_to_id
      )
      if (i + 1) % 20 == 0 or i + 1 == len(self.seq_ids):
        print(f"  loaded {i + 1}/{len(self.seq_ids)}", flush=True)

  def __len__(self) -> int:
    return len(self.seq_ids)

  def __getitem__(self, index: int) -> dict:
    seq_id = self.seq_ids[index]
    if seq_id not in self._cache:
      feat_path = self.features_root / seq_id / "features.csv"
      self._cache[seq_id] = merged_stream_from_features_csv(
        feat_path, self.object_vocab, self.object_to_id
      )
    X, y_multi, t_sec, ts_ns = self._cache[seq_id]
    return {
      "seq_id": seq_id,
      "X": torch.from_numpy(X),
      "y_multi": torch.from_numpy(y_multi),
      "length": X.shape[0],
      "t_sec": torch.from_numpy(t_sec.astype(np.float32)),
      "timestamp_ns": ts_ns,
    }


def collate_merged_seqs(batch: list[dict]) -> dict:
  lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
  t_max = int(lengths.max().item())
  vocab_size = batch[0]["X"].shape[1]
  slot_dim = batch[0]["X"].shape[2]

  X_pad = torch.full((len(batch), t_max, vocab_size, slot_dim), SENTINEL, dtype=torch.float32)
  X_pad[:, :, :, OBJ_VALID_IDX] = 0.0
  y_pad = torch.zeros((len(batch), t_max, vocab_size), dtype=torch.float32)

  for i, b in enumerate(batch):
    L = b["length"]
    X_pad[i, :L] = b["X"]
    y_pad[i, :L] = b["y_multi"]

  return {
    "X": X_pad,
    "y_multi": y_pad,
    "lengths": lengths,
    "seq_id": [b["seq_id"] for b in batch],
  }


def compute_multilabel_pos_weight(dataset: MergedSeqDataset) -> torch.Tensor:
  """BCE pos_weight per object slot (length V)."""
  vocab_size = len(dataset.object_vocab)
  name_to_slot = {name: j for j, name in enumerate(dataset.object_vocab)}
  pos = np.zeros(vocab_size, dtype=np.float64)
  neg = np.zeros(vocab_size, dtype=np.float64)
  usecols = ["timestamp_ns", "object_name", "y_interacting"]
  for seq_id in dataset.seq_ids:
    feat_path = dataset.features_root / seq_id / "features.csv"
    df = pd.read_csv(feat_path, usecols=usecols)
    df = df[df["object_name"].isin(name_to_slot)].copy()
    if df.empty:
      continue
    n_ts = int(df["timestamp_ns"].nunique())
    on = df[df["y_interacting"].astype(int) == 1]
    pos_inc = np.zeros(vocab_size, dtype=np.float64)
    if not on.empty:
      for oname, g in on.groupby("object_name", sort=False):
        slot = name_to_slot.get(str(oname))
        if slot is None:
          continue
        pos_inc[slot] = float(g["timestamp_ns"].nunique())
    pos += pos_inc
    neg += n_ts - pos_inc
  pos = np.maximum(pos, 1.0)
  neg = np.maximum(neg, 1.0)
  return torch.from_numpy((neg / pos).astype(np.float32))


@dataclass(frozen=True)
class HandStreamMeta:
  seq_id: str
  hand_side: str


class HandStreamDataset(Dataset):
  """One item per (sequence, hand) stream."""

  def __init__(
    self,
    seq_ids: list[str],
    features_root: Path,
    object_vocab: list[str] | None = None,
    object_to_id: dict[str, int] | None = None,
    hands: tuple[str, ...] = ("left", "right"),
    hand_cond: str = "onehot",
    sample_weights: dict[tuple[str, str], np.ndarray] | None = None,
    augment_train: bool = False,
    feature_jitter: float = 0.0,
    feature_channel_dropout: float = 0.0,
    rng: np.random.Generator | None = None,
  ) -> None:
    if hand_cond not in HAND_COND_CHOICES:
      raise ValueError(hand_cond)
    self.hand_cond = hand_cond
    self.sample_weights = sample_weights
    self.augment_train = augment_train
    self.feature_jitter = feature_jitter
    self.feature_channel_dropout = feature_channel_dropout
    self._rng = rng if rng is not None else np.random.default_rng()
    if object_vocab is None or object_to_id is None:
      object_vocab, object_to_id = load_object_vocab()
    self.features_root = features_root
    self.object_vocab = object_vocab
    self.object_to_id = object_to_id
    self.hands = hands
    self.items: list[tuple[str, str]] = []
    self._cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]] = {}
    for seq_id in seq_ids:
      feat_path = features_root / seq_id / "features.csv"
      if not feat_path.is_file():
        raise FileNotFoundError(feat_path)
      for hand in hands:
        self.items.append((seq_id, hand))

  def __len__(self) -> int:
    return len(self.items)

  def __getitem__(self, index: int) -> dict:
    seq_id, hand = self.items[index]
    key = (seq_id, hand, self.hand_cond)
    if key not in self._cache:
      feat_path = self.features_root / seq_id / "features.csv"
      self._cache[key] = hand_stream_from_features_csv(
        feat_path, hand, self.object_vocab, self.object_to_id, hand_cond=self.hand_cond
      )
    X, y, t_sec, ts_ns = self._cache[key]
    if self.augment_train and (self.feature_jitter > 0 or self.feature_channel_dropout > 0):
      X = augment_hand_stream_features(
        X,
        feature_jitter=self.feature_jitter,
        feature_channel_dropout=self.feature_channel_dropout,
        rng=self._rng,
      )
    out = {
      "seq_id": seq_id,
      "hand_side": hand,
      "hand_id": hand_side_to_id(hand),
      "X": torch.from_numpy(X),
      "y": torch.from_numpy(y),
      "length": X.shape[0],
      "t_sec": torch.from_numpy(t_sec.astype(np.float32)),
      "timestamp_ns": ts_ns,
    }
    if self.sample_weights is not None:
      w = self.sample_weights.get((seq_id, hand))
      if w is None:
        out["sample_w"] = torch.ones(X.shape[0], dtype=torch.float32)
      else:
        if len(w) != X.shape[0]:
          raise ValueError(
            f"sample_w length {len(w)} != stream length {X.shape[0]} for {seq_id} {hand}"
          )
        out["sample_w"] = torch.from_numpy(w.astype(np.float32))
    return out


def collate_hand_streams(batch: list[dict]) -> dict:
  lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
  t_max = int(lengths.max().item())
  vocab_size = batch[0]["X"].shape[1]
  slot_dim = batch[0]["X"].shape[2]

  X_pad = torch.full((len(batch), t_max, vocab_size, slot_dim), SENTINEL, dtype=torch.float32)
  if slot_dim >= NUM_OBJ_CHANNELS:
    X_pad[:, :, :, OBJ_VALID_IDX] = 0.0
  y_pad = torch.full((len(batch), t_max), -1, dtype=torch.long)
  has_sample_w = "sample_w" in batch[0]
  w_pad = torch.zeros((len(batch), t_max), dtype=torch.float32) if has_sample_w else None

  for i, b in enumerate(batch):
    L = b["length"]
    X_pad[i, :L] = b["X"]
    y_pad[i, :L] = b["y"]
    if w_pad is not None:
      if "sample_w" in b:
        w_pad[i, :L] = b["sample_w"]
      else:
        w_pad[i, :L] = 1.0

  hand_ids = torch.tensor([b["hand_id"] for b in batch], dtype=torch.long)

  out = {
    "X": X_pad,
    "y": y_pad,
    "lengths": lengths,
    "hand_id": hand_ids,
    "seq_id": [b["seq_id"] for b in batch],
    "hand_side": [b["hand_side"] for b in batch],
  }
  if w_pad is not None:
    out["sample_w"] = w_pad
  return out


def collapse_hand_labels(
  feat_path: Path,
  hand_side: str,
  object_to_id: dict[str, int],
) -> np.ndarray:
  """Fast path: per-timestep class ids only (no feature pivot)."""
  df = pd.read_csv(feat_path, usecols=["timestamp_ns", "hand_side", "object_name", "y_interacting"])
  hand_df = df[df["hand_side"] == hand_side]
  timestamps = sorted(hand_df["timestamp_ns"].unique())
  y = np.zeros(len(timestamps), dtype=np.int64)
  t_to_idx = {int(ts): i for i, ts in enumerate(timestamps)}
  for ts, g in hand_df.groupby("timestamp_ns", sort=False):
    ti = t_to_idx[int(ts)]
    pos = g[g["y_interacting"].astype(int) == 1]
    if len(pos) >= 1:
      oname = str(pos.iloc[0]["object_name"])
      y[ti] = object_to_id.get(oname, 0)
  return y


def compute_class_weights(
  dataset: HandStreamDataset,
  num_classes: int,
  *,
  cap: float | None = None,
) -> torch.Tensor:
  """Inverse-frequency weights for CrossEntropy (length num_classes)."""
  counts = np.zeros(num_classes, dtype=np.float64)
  for seq_id, hand in dataset.items:
    feat_path = dataset.features_root / seq_id / "features.csv"
    y = collapse_hand_labels(feat_path, hand, dataset.object_to_id)
    for c in range(num_classes):
      counts[c] += (y == c).sum()
  counts = np.maximum(counts, 1.0)
  weights = counts.sum() / (num_classes * counts)
  if cap is not None:
    weights = np.minimum(weights, cap)
  return torch.from_numpy(weights.astype(np.float32))
