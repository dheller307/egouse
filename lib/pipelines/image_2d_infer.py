"""image_2d dual-dwell inference from detector preds.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lib.eval_timeline import evaluate_timeline, pred_segments_from_pair_scores
from lib.experiments._common import fire_rate
from lib.experiments.teacher_pipeline import FROZEN_CLEANUP_CONFIG
from lib.pipelines.image_2d import load_preds_by_ts, score_sequence_from_preds
from lib.scoring.dual_dwell_config import FROZEN_DUAL_DWELL
from lib.scoring.timeline_cleanup import cleanup_pair_scores
from lib.viz.timeline import merge_hands_pair_df
from sklearn.metrics import f1_score, precision_score, recall_score


def densify_to_teacher_grid(pair_df: pd.DataFrame, teacher: pd.DataFrame) -> pd.DataFrame:
  keys = ["timestamp_ns", "hand_side", "object_name"]
  base = teacher[keys + ["t_sec"]].drop_duplicates(subset=keys)
  cols = [
    "v2_interacting",
    "iou",
    "v2_co_motion_score",
    "v2_iou_ok",
    "v2_co_motion_ok",
    "v2_frame_ok",
    "v2_iou_dwell_ms",
    "v2_co_motion_dwell_ms",
    "v2_dwell_ms",
    "box_source",
  ]
  have = [c for c in cols if c in pair_df.columns]
  merged = base.merge(pair_df[keys + have], on=keys, how="left")
  merged["v2_interacting"] = merged["v2_interacting"].fillna(0).astype(int)
  if "box_source" in merged.columns:
    merged["box_source"] = merged["box_source"].fillna("missing_det")
  else:
    merged["box_source"] = "missing_det"
  return merged


def _frame_metrics(pred: pd.DataFrame, teacher: pd.DataFrame, *, merge_hands: bool) -> dict:
  keys = ["timestamp_ns", "hand_side", "object_name"]
  pred_ref = pred[keys + ["v2_interacting"]].copy()
  teacher_ref = teacher[keys + ["v2_interacting"]].copy()
  if merge_hands:
    pred_ref = merge_hands_pair_df(pred_ref.assign(t_sec=0))
    teacher_ref = merge_hands_pair_df(teacher_ref.assign(t_sec=0))
    pred_ref = pred_ref.assign(hand_side="merged")
    teacher_ref = teacher_ref.assign(hand_side="merged")
  m = teacher_ref.merge(pred_ref, on=keys, how="left", suffixes=("_true", "_pred"))
  y_true = m["v2_interacting_true"].astype(int).to_numpy()
  y_pred = m["v2_interacting_pred"].fillna(0).astype(int).to_numpy()
  return {
    "frame_prec": float(precision_score(y_true, y_pred, zero_division=0)),
    "frame_rec": float(recall_score(y_true, y_pred, zero_division=0)),
    "frame_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "n_rows": int(len(y_true)),
    "n_positive_true": int(y_true.sum()),
    "n_positive_pred": int(y_pred.sum()),
  }


def eval_vs_teacher(
  pred: pd.DataFrame,
  teacher: pd.DataFrame,
  *,
  iou_threshold: float = 0.5,
  merge_hands: bool = False,
) -> dict:
  pred_ref = pred[["timestamp_ns", "t_sec", "hand_side", "object_name", "v2_interacting"]].copy()
  teacher_ref = teacher[
    ["timestamp_ns", "t_sec", "hand_side", "object_name", "v2_interacting"]
  ].copy()
  if merge_hands:
    pred_ref = merge_hands_pair_df(pred_ref)
    teacher_ref = merge_hands_pair_df(teacher_ref)
  frame_m = _frame_metrics(pred_ref, teacher_ref, merge_hands=merge_hands)
  tl = evaluate_timeline(
    pred_segments_from_pair_scores(teacher_ref),
    pred_segments_from_pair_scores(pred_ref),
    iou_threshold=iou_threshold,
  )
  return {
    **frame_m,
    "seg_f1": tl["seg_f1"],
    "seg_prec": tl["seg_prec"],
    "seg_rec": tl["seg_rec"],
    "edit_sim": tl["edit_sim"],
    "n_pred": int(tl.get("overall", {}).get("n_pred", 0)),
    "n_matched": int(tl.get("overall", {}).get("n_matched", 0)),
    "pair_fire_rate_pct": fire_rate(pred_ref),
  }


def run_image_2d_for_sequence(
  seq_id: str,
  *,
  preds_root: Path,
  teacher_root: Path,
  output_dir: Path,
  object_names: list[str] | None = None,
  no_cleanup: bool = False,
  merge_hands: bool = False,
  iou_threshold: float = 0.5,
) -> dict:
  """Score detector preds with image_2d dual-dwell; optionally eval vs teacher."""
  preds_path = preds_root / seq_id / "preds.jsonl"
  if not preds_path.is_file():
    alt = preds_root / "preds.jsonl"
    if alt.is_file():
      preds_path = alt
    else:
      raise FileNotFoundError(f"Missing preds.jsonl for {seq_id}: {preds_path}")

  teacher_path = teacher_root / seq_id / "pair_scores_clean.csv"
  teacher = pd.read_csv(teacher_path) if teacher_path.is_file() else None

  if object_names is None:
    if teacher is not None:
      object_names = sorted({str(n) for n in teacher["object_name"].unique()})
    else:
      raise ValueError("object_names required when teacher labels are absent")

  config = FROZEN_DUAL_DWELL.to_scoring_config()
  preds_by_ts = load_preds_by_ts(preds_path)
  if teacher is not None:
    teacher_ts = set(int(x) for x in teacher["timestamp_ns"].unique())
    preds_by_ts = {ts: rec for ts, rec in preds_by_ts.items() if ts in teacher_ts}

  raw_rows = score_sequence_from_preds(preds_by_ts, object_names, config)
  pair_df = pd.DataFrame(raw_rows)
  if pair_df.empty:
    raise RuntimeError(f"no pair rows for {seq_id}")

  if teacher is not None:
    t0 = int(teacher["timestamp_ns"].min())
    pair_df["t_sec"] = (pair_df["timestamp_ns"] - t0) / 1e9
    dense = densify_to_teacher_grid(pair_df, teacher)
  else:
    t0 = int(pair_df["timestamp_ns"].min())
    pair_df["t_sec"] = (pair_df["timestamp_ns"] - t0) / 1e9
    dense = pair_df

  if no_cleanup:
    clean = dense
    cleanup_stats = {"skipped": True}
  else:
    clean, cleanup_stats = cleanup_pair_scores(dense, FROZEN_CLEANUP_CONFIG)

  seq_out = output_dir / seq_id if teacher is not None else output_dir
  seq_out.mkdir(parents=True, exist_ok=True)
  pair_df.to_csv(seq_out / "pair_scores_raw_det.csv", index=False)
  dense.to_csv(seq_out / "pair_scores.csv", index=False)
  clean.to_csv(seq_out / "pair_scores_clean.csv", index=False)
  (seq_out / "cleanup_stats.json").write_text(json.dumps(cleanup_stats, indent=2) + "\n")
  (seq_out / "box_source.json").write_text(
    json.dumps(
      {
        "box_source": "detector_preds",
        "preds_path": str(preds_path),
        "backend_note": "100DOH hands + YOLOE objects",
        "not_hot3d_gt_boxes": True,
      },
      indent=2,
    )
    + "\n"
  )

  metrics: dict = {"seq": seq_id, "box_source": "detector_preds"}
  if teacher is not None:
    metrics.update(
      eval_vs_teacher(clean, teacher, iou_threshold=iou_threshold, merge_hands=merge_hands)
    )
    (seq_out / "summary_vs_teacher.json").write_text(json.dumps(metrics, indent=2) + "\n")
  return metrics


def eval_pred_vs_human_gt(
  pred: pd.DataFrame,
  seq_id: str,
  *,
  teacher_root: Path,
  merge_hands: bool = True,
  iou_threshold: float = 0.5,
) -> dict:
  """Pair-level metrics vs human GT segments on the teacher (ts, hand, object) grid."""
  from lib.clip import mapping_path_for_sequence, segments_path_for_sequence
  from lib.paths import EGOUSE_ROOT, HOT3D_DATA_DIR
  from lib.viz.timeline import build_timeline_table, gt_pair_df_from_timeline

  seq_path = HOT3D_DATA_DIR / seq_id
  gt_path = segments_path_for_sequence(seq_path, EGOUSE_ROOT)
  mapping_path = mapping_path_for_sequence(seq_path)
  frame_path = teacher_root / seq_id / "frame_scores_clean.csv"
  teacher_path = teacher_root / seq_id / "pair_scores_clean.csv"
  if not gt_path.is_file():
    raise FileNotFoundError(gt_path)
  timeline_df = build_timeline_table(frame_path, gt_path, mapping_path)
  teacher = pd.read_csv(teacher_path)
  gt_pair = gt_pair_df_from_timeline(teacher, timeline_df)
  metrics = eval_vs_teacher(pred, gt_pair, iou_threshold=iou_threshold, merge_hands=merge_hands)
  metrics["seq"] = seq_id
  metrics["reference"] = "human_gt"
  metrics["merge_hands"] = bool(merge_hands)
  return metrics
