"""Load GRU checkpoint and expand per-hand class predictions to pair_scores rows."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lib.student_gru_model import StudentHandGRU
from lib.student_sequence_data import (
  SLOT_DIM_ONEHOT,
  hand_side_to_id,
  hand_stream_from_features_csv,
  merged_stream_from_features_csv,
  slot_dim_for_hand_cond,
)


def load_gru_checkpoint(ckpt_path: Path, device: torch.device | None = None) -> tuple[StudentHandGRU, list[str], dict[str, int], dict]:
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  hparams = ckpt["hparams"]
  hand_cond = hparams.get("hand_cond", "onehot")
  label_mode = hparams.get("label_mode", "exclusive")
  model = StudentHandGRU(
    vocab_size=hparams["vocab_size"],
    slot_dim=hparams.get("slot_dim", SLOT_DIM_ONEHOT),
    d_slot=hparams["d_slot"],
    hidden=hparams["hidden"],
    num_layers=hparams["num_layers"],
    dropout=hparams["dropout"],
    num_classes=hparams["num_classes"],
    hand_cond=hand_cond,
    hand_embed_dim=int(hparams.get("hand_embed_dim", 8)),
    label_mode=label_mode,
  )
  model.load_state_dict(ckpt["model_state"])
  model.to(device)
  model.eval()
  vocab = ckpt["object_vocab"]
  obj_to_id = ckpt["object_to_id"]
  return model, vocab, obj_to_id, hparams


def decode_exclusive_classes(
  prob: np.ndarray,
  *,
  decode_margin: float = 0.0,
  decode_min_prob: float = 0.0,
) -> np.ndarray:
  """
  Exclusive per-hand decode with optional margin over background (class 0).

  Accept class c > 0 only if prob[c] >= prob[0] + margin and prob[c] >= min_prob.
  """
  prob = np.asarray(prob, dtype=np.float64)
  t_len, num_classes = prob.shape
  pred = np.zeros(t_len, dtype=np.int64)
  bg_prob = prob[:, 0]
  for t in range(t_len):
    best_c = 0
    best_p = bg_prob[t]
    for c in range(1, num_classes):
      p = prob[t, c]
      if p > best_p:
        best_p = p
        best_c = c
    if best_c > 0:
      if best_p >= bg_prob[t] + decode_margin and best_p >= decode_min_prob:
        pred[t] = best_c
  return pred


@torch.no_grad()
def predict_hand_stream(
  model: StudentHandGRU,
  X: np.ndarray,
  device: torch.device,
  *,
  hand_id: int,
  decode_margin: float = 0.0,
  decode_min_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
  """X (T, V, slot_dim) -> class_ids (T,), probs (T, num_classes)."""
  xt = torch.from_numpy(X).unsqueeze(0).to(device)
  lengths = torch.tensor([X.shape[0]], dtype=torch.long, device=device)
  hid = torch.tensor([hand_id], dtype=torch.long, device=device)
  logits = model(xt, lengths, hand_id=hid if model.hand_embed is not None else None)[0]
  if model.label_mode == "multilabel":
    prob = torch.sigmoid(logits).cpu().numpy()
    pred = (prob >= 0.5).astype(np.int64)
    return pred, prob
  prob = torch.softmax(logits, dim=-1).cpu().numpy()
  if decode_margin > 0 or decode_min_prob > 0:
    pred = decode_exclusive_classes(
      prob, decode_margin=decode_margin, decode_min_prob=decode_min_prob
    )
  else:
    pred = prob.argmax(axis=-1)
  return pred, prob


@torch.no_grad()
def predict_merged_stream(
  model: StudentHandGRU,
  X: np.ndarray,
  device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
  """X (T, V, slot_dim) -> on_multi (T, V), probs (T, V)."""
  xt = torch.from_numpy(X).unsqueeze(0).to(device)
  lengths = torch.tensor([X.shape[0]], dtype=torch.long, device=device)
  logits = model(xt, lengths, hand_id=None)[0]
  prob = torch.sigmoid(logits).cpu().numpy()
  on = (prob >= 0.5).astype(np.int64)
  return on, prob


def pair_scores_merged_for_sequence(
  seq_id: str,
  features_root: Path,
  model: StudentHandGRU,
  object_vocab: list[str],
  object_to_id: dict[str, int],
  device: torch.device,
  *,
  prob_thresh: float = 0.5,
) -> pd.DataFrame:
  """Long pair_scores with hand_side=merged; multiple objects may be ON at one timestamp."""
  feat_path = features_root / seq_id / "features.csv"
  feat = pd.read_csv(feat_path)
  X, _y, t_sec, ts_ns = merged_stream_from_features_csv(
    feat_path, object_vocab, object_to_id
  )
  on_multi, prob = predict_merged_stream(model, X, device)
  if prob_thresh != 0.5:
    on_multi = (prob >= prob_thresh).astype(np.int64)

  rows: list[dict] = []
  for ti, ts in enumerate(ts_ns):
    for j, oname in enumerate(object_vocab):
      rows.append(
        {
          "timestamp_ns": ts,
          "t_sec": float(t_sec[ti]),
          "hand_side": "merged",
          "object_name": oname,
          "v2_interacting": int(on_multi[ti, j]),
          "student_prob": float(prob[ti, j]),
        }
      )
  out = pd.DataFrame(rows)
  # Keep only rows that appear in features long format for eval join (optional trim)
  keys = feat[["timestamp_ns", "object_name"]].drop_duplicates()
  out = out.merge(keys, on=["timestamp_ns", "object_name"], how="inner")
  out["hand_side"] = "merged"
  out = out.sort_values(["timestamp_ns", "object_name"]).reset_index(drop=True)
  return out


def pair_scores_for_sequence(
  seq_id: str,
  features_root: Path,
  model: StudentHandGRU,
  object_vocab: list[str],
  object_to_id: dict[str, int],
  device: torch.device,
  *,
  hand_cond: str | None = None,
  decode_margin: float = 0.0,
  decode_min_prob: float = 0.0,
) -> pd.DataFrame:
  feat_path = features_root / seq_id / "features.csv"
  feat = pd.read_csv(feat_path)
  if hand_cond is None:
    hand_cond = model.hand_cond
  if model.label_mode == "multilabel":
    return pair_scores_merged_for_sequence(
      seq_id, features_root, model, object_vocab, object_to_id, device
    )

  pred_frames: list[pd.DataFrame] = []
  for hand in ("left", "right"):
    sub_hand = feat[feat["hand_side"] == hand]
    if sub_hand.empty:
      continue
    X, _y, _t_sec, ts_ns = hand_stream_from_features_csv(
      feat_path, hand, object_vocab, object_to_id, hand_cond=hand_cond
    )
    pred_cls, prob = predict_hand_stream(
      model,
      X,
      device,
      hand_id=hand_side_to_id(hand),
      decode_margin=decode_margin,
      decode_min_prob=decode_min_prob,
    )
    ts_to_pred = {ts_ns[i]: (int(pred_cls[i]), prob[i]) for i in range(len(ts_ns))}

    rows = []
    for _, row in sub_hand.iterrows():
      ts = int(row["timestamp_ns"])
      oname = str(row["object_name"])
      cls_id, pvec = ts_to_pred.get(ts, (0, None))
      oid = object_to_id.get(oname, -1)
      is_on = int(cls_id == oid and oid > 0)
      if pvec is not None and oid > 0:
        student_prob = float(pvec[cls_id]) if is_on else float(pvec[oid])
      else:
        student_prob = 0.0
      rows.append(
        {
          "timestamp_ns": ts,
          "t_sec": float(row["t_sec"]),
          "hand_side": hand,
          "object_name": oname,
          "v2_interacting": is_on,
          "student_prob": student_prob,
        }
      )
    pred_frames.append(pd.DataFrame(rows))

  if not pred_frames:
    return pd.DataFrame()
  out = pd.concat(pred_frames, ignore_index=True)
  return out.sort_values(["timestamp_ns", "hand_side", "object_name"]).reset_index(drop=True)


def benchmark_pair_scores_cpu(
  seq_id: str,
  checkpoint: Path,
  features_root: Path,
  *,
  n_warmup: int = 2,
  n_repeats: int = 5,
  warn_ms_per_frame: float = 33.0,
) -> dict:
  """Wall-clock CPU inference for pair_scores_for_sequence (median ms per frame)."""
  device = torch.device("cpu")
  model, vocab, obj_to_id, _ = load_gru_checkpoint(checkpoint, device)
  feat_path = features_root / seq_id / "features.csv"
  n_frames = int(pd.read_csv(feat_path, usecols=["timestamp_ns"])["timestamp_ns"].nunique())

  def _run() -> float:
    t0 = time.perf_counter()
    pair_scores_for_sequence(seq_id, features_root, model, vocab, obj_to_id, device)
    return time.perf_counter() - t0

  for _ in range(n_warmup):
    _run()
  elapsed = [_run() for _ in range(n_repeats)]
  median_s = float(np.median(elapsed))
  ms_per_frame = (median_s / max(n_frames, 1)) * 1000.0
  out = {
    "seq_id": seq_id,
    "checkpoint": str(checkpoint),
    "device": "cpu",
    "n_frames": n_frames,
    "median_wall_s": median_s,
    "median_ms_per_frame": ms_per_frame,
    "warn_ms_per_frame_threshold": warn_ms_per_frame,
    "over_threshold": ms_per_frame > warn_ms_per_frame,
  }
  return out


def write_cpu_infer_bench(path: Path, bench: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(bench, indent=2))
  if bench.get("over_threshold"):
    print(
      f"WARNING: CPU infer {bench['median_ms_per_frame']:.2f} ms/frame exceeds "
      f"{bench['warn_ms_per_frame_threshold']} ms/frame",
      flush=True,
    )
