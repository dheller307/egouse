"""Checkpoint selection on prove stays per-hand (not L∪R merged).

Public tables use ``src/eval_student_gru.py --merge-hands``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from lib.eval_timeline import evaluate_timeline, pred_segments_from_pair_scores
from lib.student_gru_infer import pair_scores_for_sequence
from lib.student_gru_model import StudentHandGRU


CHECKPOINT_METRICS = frozenset({
  "val_ce",
  "prove_timeline",
  "prove_prec",
  "prove_f05",
  "prove_prec_penalized",
})


def combine_prove_score(
  mean_seg_f1: float,
  mean_edit_sim: float,
  *,
  seg_weight: float = 3.0,
  edit_weight: float = 1.0,
) -> float:
  """Weighted prove objective; default balances seg vs edit (~equal $ at baseline)."""
  return seg_weight * mean_seg_f1 + edit_weight * mean_edit_sim


def compute_spur_ratio(n_pred: int, n_matched: int) -> float:
  """Fraction of predicted segments that are spurious (unmatched)."""
  return (n_pred - n_matched) / max(n_pred, 1)


def compute_f05(prec: float, rec: float) -> float:
  """F0.5 emphasizing precision."""
  denom = 0.25 * prec + rec
  if denom <= 0:
    return 0.0
  return (1.25 * prec * rec) / denom


def prove_prec_penalized(mean_seg_prec: float, spur_ratio: float, *, penalty: float = 0.15) -> float:
  return mean_seg_prec - penalty * spur_ratio


def score_for_checkpoint(
  metric: str,
  metrics: dict[str, float],
  *,
  spur_penalty: float = 0.15,
) -> float:
  """Map prove metrics dict to scalar checkpoint score (higher is better)."""
  if metric == "prove_timeline":
    return float(metrics.get("combined_timeline_score", float("nan")))
  if metric == "prove_prec":
    return float(metrics.get("mean_seg_prec", float("nan")))
  if metric == "prove_f05":
    return float(metrics.get("prove_f05", float("nan")))
  if metric == "prove_prec_penalized":
    return prove_prec_penalized(
      float(metrics.get("mean_seg_prec", 0.0)),
      float(metrics.get("spur_ratio", 0.0)),
      penalty=spur_penalty,
    )
  raise ValueError(f"not a prove checkpoint metric: {metric}")


@dataclass
class ProveMetrics:
  mean_seg_f1: float
  mean_edit_sim: float
  mean_seg_prec: float
  mean_seg_rec: float
  spur_ratio: float
  prove_f05: float
  prove_prec_penalized: float
  combined_timeline_score: float
  n_pred_total: int
  n_matched_total: int
  mean_frame_prec: float
  mean_frame_rec: float
  mean_pred_on_rate: float
  mean_teacher_on_rate: float

  def as_dict(self) -> dict[str, float]:
    return {
      "mean_seg_f1": self.mean_seg_f1,
      "mean_edit_sim": self.mean_edit_sim,
      "mean_seg_prec": self.mean_seg_prec,
      "mean_seg_rec": self.mean_seg_rec,
      "spur_ratio": self.spur_ratio,
      "prove_f05": self.prove_f05,
      "prove_prec_penalized": self.prove_prec_penalized,
      "combined_timeline_score": self.combined_timeline_score,
      "n_pred_total": float(self.n_pred_total),
      "n_matched_total": float(self.n_matched_total),
      "mean_frame_prec": self.mean_frame_prec,
      "mean_frame_rec": self.mean_frame_rec,
      "mean_pred_on_rate": self.mean_pred_on_rate,
      "mean_teacher_on_rate": self.mean_teacher_on_rate,
    }


def _frame_metrics_from_pairs(student: pd.DataFrame, feat: pd.DataFrame) -> dict[str, float]:
  keys = ["timestamp_ns", "hand_side", "object_name"]
  m = feat[keys + ["y_interacting"]].merge(
    student[keys + ["v2_interacting"]],
    on=keys,
    how="left",
    suffixes=("_true", "_pred"),
  )
  y_true = m["y_interacting"].astype(int).to_numpy()
  y_pred = m["v2_interacting"].fillna(0).astype(int).to_numpy()
  tp = int(((y_true == 1) & (y_pred == 1)).sum())
  fp = int(((y_true == 0) & (y_pred == 1)).sum())
  fn = int(((y_true == 1) & (y_pred == 0)).sum())
  prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
  rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  return {"frame_prec": float(prec), "frame_rec": float(rec)}


@torch.no_grad()
def prove_full_metrics(
  model: StudentHandGRU,
  object_vocab: list[str],
  object_to_id: dict[str, int],
  device: torch.device,
  seq_ids: list[str],
  features_root: Path,
  teacher_root: Path,
  *,
  iou_threshold: float = 0.5,
  seg_weight: float = 3.0,
  edit_weight: float = 1.0,
  spur_penalty: float = 0.15,
  decode_margin: float = 0.0,
  decode_min_prob: float = 0.0,
) -> ProveMetrics:
  """Aggregate prove timeline + precision metrics; no disk writes."""
  nan = float("nan")
  if not seq_ids:
    return ProveMetrics(nan, nan, nan, nan, nan, nan, nan, nan, 0, 0, nan, nan, nan, nan)

  seg_f1s: list[float] = []
  edits: list[float] = []
  seg_precs: list[float] = []
  seg_recs: list[float] = []
  frame_precs: list[float] = []
  frame_recs: list[float] = []
  pred_on_rates: list[float] = []
  teacher_on_rates: list[float] = []
  n_pred_total = 0
  n_matched_total = 0

  for seq_id in seq_ids:
    feat_path = features_root / seq_id / "features.csv"
    teacher_path = teacher_root / seq_id / "pair_scores_clean.csv"
    if not feat_path.is_file() or not teacher_path.is_file():
      continue
    feat = pd.read_csv(feat_path)
    teacher = pd.read_csv(teacher_path)
    teacher_ref = teacher[
      ["timestamp_ns", "t_sec", "hand_side", "object_name", "v2_interacting"]
    ]
    student = pair_scores_for_sequence(
      seq_id,
      features_root,
      model,
      object_vocab,
      object_to_id,
      device,
      decode_margin=decode_margin,
      decode_min_prob=decode_min_prob,
    )
    pred_segs = pred_segments_from_pair_scores(student)
    gt_segs = pred_segments_from_pair_scores(teacher_ref)
    tl = evaluate_timeline(gt_segs, pred_segs, iou_threshold=iou_threshold)
    seg_f1s.append(float(tl["seg_f1"]))
    edits.append(float(tl["edit_sim"]))
    seg_precs.append(float(tl["seg_prec"]))
    seg_recs.append(float(tl["seg_rec"]))
    n_pred_total += int(tl.get("overall", {}).get("n_pred", 0))
    n_matched_total += int(tl.get("overall", {}).get("n_matched", 0))
    fm = _frame_metrics_from_pairs(student, feat)
    frame_precs.append(fm["frame_prec"])
    frame_recs.append(fm["frame_rec"])
    y_true = feat["y_interacting"].astype(int)
    pred_on = student.merge(
      feat[["timestamp_ns", "hand_side", "object_name"]],
      on=["timestamp_ns", "hand_side", "object_name"],
      how="right",
    )["v2_interacting"].fillna(0).astype(int)
    pred_on_rates.append(float(pred_on.sum()) / max(len(y_true), 1))
    teacher_on_rates.append(float(y_true.sum()) / max(len(y_true), 1))

  if not seg_f1s:
    return ProveMetrics(nan, nan, nan, nan, nan, nan, nan, nan, 0, 0, nan, nan, nan, nan)

  mean_seg = sum(seg_f1s) / len(seg_f1s)
  mean_edit = sum(edits) / len(edits)
  mean_prec = sum(seg_precs) / len(seg_precs)
  mean_rec = sum(seg_recs) / len(seg_recs)
  spur = compute_spur_ratio(n_pred_total, n_matched_total)
  f05 = compute_f05(mean_prec, mean_rec)
  penalized = prove_prec_penalized(mean_prec, spur, penalty=spur_penalty)
  combined = combine_prove_score(mean_seg, mean_edit, seg_weight=seg_weight, edit_weight=edit_weight)

  return ProveMetrics(
    mean_seg_f1=mean_seg,
    mean_edit_sim=mean_edit,
    mean_seg_prec=mean_prec,
    mean_seg_rec=mean_rec,
    spur_ratio=spur,
    prove_f05=f05,
    prove_prec_penalized=penalized,
    combined_timeline_score=combined,
    n_pred_total=n_pred_total,
    n_matched_total=n_matched_total,
    mean_frame_prec=sum(frame_precs) / len(frame_precs),
    mean_frame_rec=sum(frame_recs) / len(frame_recs),
    mean_pred_on_rate=sum(pred_on_rates) / len(pred_on_rates),
    mean_teacher_on_rate=sum(teacher_on_rates) / len(teacher_on_rates),
  )


@torch.no_grad()
def prove_timeline_score(
  model: StudentHandGRU,
  object_vocab: list[str],
  object_to_id: dict[str, int],
  device: torch.device,
  seq_ids: list[str],
  features_root: Path,
  teacher_root: Path,
  *,
  iou_threshold: float = 0.5,
  seg_weight: float = 3.0,
  edit_weight: float = 1.0,
  spur_penalty: float = 0.15,
  decode_margin: float = 0.0,
  decode_min_prob: float = 0.0,
) -> tuple[float, float, float]:
  """Mean per-hand seg_f1 and edit_sim on prove sequences; no disk writes.

  Checkpoint selection stays **per-hand** (not L∪R merged). Public tables use
  `src/eval_student_gru.py --merge-hands`.

  Returns (mean_seg_f1, mean_edit_sim, combined_score).
  combined_score = seg_weight * mean_seg_f1 + edit_weight * mean_edit_sim
  """
  m = prove_full_metrics(
    model,
    object_vocab,
    object_to_id,
    device,
    seq_ids,
    features_root,
    teacher_root,
    iou_threshold=iou_threshold,
    seg_weight=seg_weight,
    edit_weight=edit_weight,
    spur_penalty=spur_penalty,
    decode_margin=decode_margin,
    decode_min_prob=decode_min_prob,
  )
  return m.mean_seg_f1, m.mean_edit_sim, m.combined_timeline_score
