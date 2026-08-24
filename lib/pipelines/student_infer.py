"""GRU student inference: features.csv → pair_scores.csv."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lib.paths import DEFAULT_STUDENT_CKPT, EGOUSE_ROOT
from lib.student_features import FEATURE_COLUMNS, ROLL_K, ROLL_MIN_VALID

DEFAULT_CKPT = DEFAULT_STUDENT_CKPT


def infer_pair_scores(
  seq_id: str,
  features_root: Path,
  checkpoint: Path,
  *,
  device=None,
  decode_margin: float = 0.0,
  decode_min_prob: float = 0.0,
) -> pd.DataFrame:
  import torch

  from lib.student_gru_infer import load_gru_checkpoint, pair_scores_for_sequence

  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model, vocab, obj_to_id, _ = load_gru_checkpoint(checkpoint, device)
  return pair_scores_for_sequence(
    seq_id,
    features_root,
    model,
    vocab,
    obj_to_id,
    device,
    decode_margin=decode_margin,
    decode_min_prob=decode_min_prob,
  )


def write_student_output(
  pair_scores: pd.DataFrame,
  output_dir: Path,
  *,
  seq_id: str,
  checkpoint: Path,
  extra_meta: dict | None = None,
) -> Path:
  output_dir.mkdir(parents=True, exist_ok=True)
  out_csv = output_dir / "pair_scores.csv"
  pair_scores.to_csv(out_csv, index=False)
  meta = {
    "seq": seq_id,
    "checkpoint": str(checkpoint),
    "n_rows": int(len(pair_scores)),
    "n_on": int(pair_scores["v2_interacting"].astype(bool).sum()) if len(pair_scores) else 0,
  }
  if extra_meta:
    meta.update(extra_meta)
  (output_dir / "infer_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
  return out_csv


def _best_hand_for_side(hands: list[dict], side: str) -> dict | None:
  sided = [h for h in hands if h.get("side") == side]
  if sided:
    return max(sided, key=lambda h: float(h.get("score") or 0.0))
  return max(hands, key=lambda h: float(h.get("score") or 0.0)) if hands else None


def _best_obj_for_name(objects: list[dict], object_name: str) -> dict | None:
  named = [o for o in objects if o.get("name") == object_name]
  if named:
    return max(named, key=lambda o: float(o.get("score") or 0.0))
  return None


def build_features_from_teacher_grid(
  preds_by_ts: dict[int, dict],
  teacher_clean: pd.DataFrame,
  *,
  roll_k: int = ROLL_K,
) -> pd.DataFrame:
  """Build features.csv joined to teacher grid (HOT3D eval path)."""
  from src.build_student_features import _add_temporal, _pair_static

  rows = []
  for _, tr in teacher_clean.iterrows():
    ts = int(tr["timestamp_ns"])
    pred = preds_by_ts.get(ts)
    if pred is None:
      continue
    hand_side = str(tr["hand_side"])
    object_name = str(tr["object_name"])
    hand = _best_hand_for_side(pred.get("hands") or [], hand_side)
    obj = _best_obj_for_name(pred.get("objects") or [], object_name)
    static = _pair_static(hand, obj)
    t_sec = float(tr["t_sec"]) if "t_sec" in tr and tr["t_sec"] == tr["t_sec"] else float("nan")
    rows.append(
      {
        "timestamp_ns": ts,
        "t_sec": t_sec,
        "hand_side": hand_side,
        "object_name": object_name,
        **static,
        "y_interacting": int(tr.get("v2_interacting") or 0),
      }
    )
  out = pd.DataFrame(rows)
  if len(out):
    out = _add_temporal(out, roll_k)
  keep = ["timestamp_ns", "t_sec", "hand_side", "object_name"] + FEATURE_COLUMNS + ["y_interacting"]
  return out[[c for c in keep if c in out.columns]]


def build_features_deploy_grid(
  preds_by_ts: dict[int, dict],
  object_names: list[str],
  *,
  roll_k: int = ROLL_K,
) -> pd.DataFrame:
  """Build features for deploy (no teacher labels): all ts × L/R × objects."""
  from src.build_student_features import _add_temporal, _pair_static

  timestamps = sorted(preds_by_ts.keys())
  rows = []
  for i, ts in enumerate(timestamps):
    pred = preds_by_ts[ts]
    t_sec = float(i)  # relative index; no HOT3D timecode
    for hand_side in ("left", "right"):
      for object_name in object_names:
        hand = _best_hand_for_side(pred.get("hands") or [], hand_side)
        obj = _best_obj_for_name(pred.get("objects") or [], object_name)
        static = _pair_static(hand, obj)
        rows.append(
          {
            "timestamp_ns": ts,
            "t_sec": t_sec,
            "hand_side": hand_side,
            "object_name": object_name,
            **static,
            "y_interacting": 0,
          }
        )
  out = pd.DataFrame(rows)
  if len(out):
    out = _add_temporal(out, roll_k)
  keep = ["timestamp_ns", "t_sec", "hand_side", "object_name"] + FEATURE_COLUMNS + ["y_interacting"]
  return out[[c for c in keep if c in out.columns]]
