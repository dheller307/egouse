"""Timeline-level eval: segment F1 @ tIoU and Edit (matches swimlane plot)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimelineSegment:
  hand_side: str
  object_name: str
  t_start: float
  t_end: float

  @property
  def duration(self) -> float:
    return max(0.0, self.t_end - self.t_start)


def on_intervals(
  t: np.ndarray,
  on: np.ndarray,
  *,
  max_gap_factor: float = 2.0,
) -> list[tuple[float, float]]:
  """
  Merge consecutive True samples into [t_start, t_end) intervals.

  Same rule as the per-object swimlane plot (fill / dashed bars).

  Gap-aware: if two successive samples are farther apart than
  ``max_gap_factor * median_dt``, treat the stream as broken — close the
  current ON at ``last_on_t + median_dt`` instead of spanning a hole in a
  sparse per-object timeseries (pair_scores often omit off-screen objects).
  """
  intervals: list[tuple[float, float]] = []
  if len(t) == 0:
    return intervals
  on = np.asarray(on, dtype=bool)
  t = np.asarray(t, dtype=float)
  if len(t) > 1:
    diffs = np.diff(t)
    positive = diffs[diffs > 0]
    median_dt = float(np.median(positive)) if len(positive) else 0.033
  else:
    median_dt = 0.033
  median_dt = max(median_dt, 1e-6)
  max_dt = max_gap_factor * median_dt

  start: float | None = None
  last_on_t: float | None = None
  for i, flag in enumerate(on):
    ti = float(t[i])
    if start is not None and last_on_t is not None and (ti - last_on_t) > max_dt:
      # Sparse hole: close before advancing.
      intervals.append((start, last_on_t + median_dt))
      start = None
      last_on_t = None
    if flag:
      if start is None:
        start = ti
      last_on_t = ti
    elif start is not None:
      intervals.append((start, ti))
      start = None
      last_on_t = None
  if start is not None and last_on_t is not None:
    intervals.append((start, last_on_t + median_dt))
  return intervals


def temporal_iou(a: TimelineSegment, b: TimelineSegment) -> float:
  if a.hand_side != b.hand_side or a.object_name != b.object_name:
    return 0.0
  inter = max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))
  union = max(a.t_end, b.t_end) - min(a.t_start, b.t_start)
  if union <= 0:
    return 0.0
  return inter / union


def pred_segments_from_pair_scores(pair_df: pd.DataFrame) -> list[TimelineSegment]:
  """Predicted swimlane bars: (hand, object) contiguous v2_interacting runs."""
  required = {"hand_side", "object_name", "t_sec", "v2_interacting"}
  missing = required - set(pair_df.columns)
  if missing:
    raise KeyError(f"pair_scores missing columns: {sorted(missing)}")

  segments: list[TimelineSegment] = []
  for (hand, obj), sub in pair_df.groupby(["hand_side", "object_name"], sort=False):
    sub = sub.sort_values("t_sec")
    t = sub["t_sec"].to_numpy(dtype=float)
    on = sub["v2_interacting"].astype(bool).to_numpy()
    for t0, t1 in on_intervals(t, on):
      if t1 > t0:
        segments.append(
          TimelineSegment(str(hand), str(obj), float(t0), float(t1))
        )
  return segments


def gt_segments_from_timeline_df(timeline_df: pd.DataFrame) -> list[TimelineSegment]:
  """
  GT dashed bars as drawn: contiguous frames with the same gt_object_name.

  Uses labeled frame rows (same source as timeline_per_object.png dashed boxes).
  """
  required = {"hand_side", "t_sec", "gt_object_name", "gt_interacting"}
  missing = required - set(timeline_df.columns)
  if missing:
    raise KeyError(f"timeline_df missing columns: {sorted(missing)}")

  segments: list[TimelineSegment] = []
  for hand, hand_df in timeline_df.groupby("hand_side", sort=False):
    hand_df = hand_df.sort_values("t_sec").reset_index(drop=True)
    t = hand_df["t_sec"].to_numpy(dtype=float)
    names = hand_df["gt_object_name"].astype(str).to_numpy()
    interacting = hand_df["gt_interacting"].astype(bool).to_numpy()
    for obj in sorted(set(names) - {"none"}):
      on = interacting & (names == obj)
      for t0, t1 in on_intervals(t, on):
        if t1 > t0:
          segments.append(
            TimelineSegment(str(hand), str(obj), float(t0), float(t1))
          )
  return segments


def segment_f1_at_iou(
  gt_segments: list[TimelineSegment],
  pred_segments: list[TimelineSegment],
  *,
  iou_threshold: float = 0.5,
) -> dict[str, float | int]:
  """
  One-to-one matching at temporal IoU >= threshold (same hand + object).

  Greedy: highest IoU pairs first.
  """
  n_gt = len(gt_segments)
  n_pred = len(pred_segments)
  if n_gt == 0 and n_pred == 0:
    return {
      "n_gt": 0,
      "n_pred": 0,
      "n_matched": 0,
      "seg_prec": 1.0,
      "seg_rec": 1.0,
      "seg_f1": 1.0,
      "iou_threshold": iou_threshold,
    }

  pairs: list[tuple[float, int, int]] = []
  for i, g in enumerate(gt_segments):
    for j, p in enumerate(pred_segments):
      iou = temporal_iou(g, p)
      if iou >= iou_threshold:
        pairs.append((iou, i, j))
  pairs.sort(reverse=True)

  used_gt: set[int] = set()
  used_pred: set[int] = set()
  matched = 0
  for _iou, i, j in pairs:
    if i in used_gt or j in used_pred:
      continue
    used_gt.add(i)
    used_pred.add(j)
    matched += 1

  prec = matched / n_pred if n_pred else 0.0
  rec = matched / n_gt if n_gt else 0.0
  f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
  return {
    "n_gt": n_gt,
    "n_pred": n_pred,
    "n_matched": matched,
    "seg_prec": round(prec, 3),
    "seg_rec": round(rec, 3),
    "seg_f1": round(f1, 3),
    "iou_threshold": iou_threshold,
  }


def _segment_label_sequence(segments: list[TimelineSegment]) -> list[str]:
  """Time-ordered labels for Edit (hand|object)."""
  ordered = sorted(segments, key=lambda s: (s.t_start, s.t_end, s.object_name))
  return [f"{s.hand_side}|{s.object_name}" for s in ordered]


def _levenshtein(a: list[str], b: list[str]) -> int:
  if not a:
    return len(b)
  if not b:
    return len(a)
  prev = list(range(len(b) + 1))
  for i, ca in enumerate(a, start=1):
    cur = [i]
    for j, cb in enumerate(b, start=1):
      ins = cur[j - 1] + 1
      delete = prev[j] + 1
      sub = prev[j - 1] + (0 if ca == cb else 1)
      cur.append(min(ins, delete, sub))
    prev = cur
  return prev[-1]


def edit_score(
  gt_segments: list[TimelineSegment],
  pred_segments: list[TimelineSegment],
) -> dict[str, float | int]:
  """
  Normalized Edit similarity on time-ordered (hand|object) segment labels.

  edit_sim = 1 - levenshtein / max(len_gt, len_pred)  (1 = perfect).
  Uses the same interval lists as segment F1 (swimlane bars), not per-frame rows.
  """
  gt_seq = _segment_label_sequence(gt_segments)
  pred_seq = _segment_label_sequence(pred_segments)
  dist = _levenshtein(gt_seq, pred_seq)
  denom = max(len(gt_seq), len(pred_seq))
  sim = 1.0 if denom == 0 else 1.0 - dist / denom
  return {
    "n_gt_labels": len(gt_seq),
    "n_pred_labels": len(pred_seq),
    "edit_distance": dist,
    "edit_sim": round(sim, 3),
  }


def evaluate_timeline(
  gt_segments: list[TimelineSegment],
  pred_segments: list[TimelineSegment],
  *,
  iou_threshold: float = 0.5,
) -> dict:
  """Primary: seg_f1 @ iou; also edit_sim (+ breakdowns per hand)."""
  overall_f1 = segment_f1_at_iou(gt_segments, pred_segments, iou_threshold=iou_threshold)
  overall_edit = edit_score(gt_segments, pred_segments)

  hands = sorted({s.hand_side for s in gt_segments} | {s.hand_side for s in pred_segments})
  per_hand = {}
  for hand in hands:
    g = [s for s in gt_segments if s.hand_side == hand]
    p = [s for s in pred_segments if s.hand_side == hand]
    per_hand[hand] = {
      **segment_f1_at_iou(g, p, iou_threshold=iou_threshold),
      **edit_score(g, p),
    }

  return {
    "primary": "seg_f1",
    "seg_f1": overall_f1["seg_f1"],
    "seg_prec": overall_f1["seg_prec"],
    "seg_rec": overall_f1["seg_rec"],
    "edit_sim": overall_edit["edit_sim"],
    "edit_distance": overall_edit["edit_distance"],
    "iou_threshold": iou_threshold,
    "overall": {**overall_f1, **overall_edit},
    "per_hand": per_hand,
  }
