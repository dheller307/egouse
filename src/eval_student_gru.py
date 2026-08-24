#!/usr/bin/env python3
"""Evaluate GRU student vs teacher y*_clean timelines.

Default follows checkpoint hparams (per-hand for phase_d_r7). Pass ``--merge-hands``
for public tables: L∪R merge of **both** student and teacher before frame and
segment metrics (same rule as swimlane plots).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.eval_timeline import evaluate_timeline, pred_segments_from_pair_scores
from lib.pipelines.image_2d_infer import eval_vs_teacher
from lib.student_gru_prove_eval import (
  compute_f05,
  compute_spur_ratio,
  prove_prec_penalized,
  combine_prove_score,
)
from lib.paths import DEFAULT_STUDENT_CKPT, EGOUSE_ROOT
from lib.student_gru_infer import load_gru_checkpoint, pair_scores_for_sequence
from lib.student_sequence_data import load_seq_list
from lib.viz.timeline import merge_hands_pair_df

DEFAULT_FEATURES_ROOT = EGOUSE_ROOT / "outputs" / "student" / "features_raw"
DEFAULT_TEACHER_ROOT = EGOUSE_ROOT / "outputs" / "teacher"
DEFAULT_CKPT = DEFAULT_STUDENT_CKPT


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Eval GRU student vs teacher timelines")
  p.add_argument("--model", type=Path, default=DEFAULT_CKPT)
  p.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
  p.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
  p.add_argument(
    "--seq-list",
    type=Path,
    default=EGOUSE_ROOT / "config" / "seq_lists" / "prove.txt",
  )
  p.add_argument("--output-dir", type=Path, default=EGOUSE_ROOT / "outputs" / "student" / "gru_v1_onehot")
  p.add_argument("--iou-threshold", type=float, default=0.5)
  p.add_argument("--decode-margin", type=float, default=0.0)
  p.add_argument("--decode-min-prob", type=float, default=0.0)
  p.add_argument("--prove-spur-penalty", type=float, default=0.15)
  p.add_argument("--device", type=str, default="auto")
  p.add_argument(
    "--merge-hands",
    action="store_true",
    help="Eval after L∪R merge (ON if either hand ON per timestamp×object). "
    "Applies to both student and teacher before frame and segment metrics.",
  )
  p.add_argument(
    "--student-pairs-root",
    type=Path,
    default=None,
    help="Reuse pair_scores_student.csv from <root>/prove/<seq>/ (skip model inference).",
  )
  return p.parse_args()


def _frame_metrics_from_pairs(student: pd.DataFrame, feat: pd.DataFrame) -> dict:
  keys = ["timestamp_ns", "hand_side", "object_name"]
  m = feat[keys + ["y_interacting"]].merge(
    student[keys + ["v2_interacting"]],
    on=keys,
    how="left",
    suffixes=("_true", "_pred"),
  )
  y_true = m["y_interacting"].astype(int).to_numpy()
  y_pred = m["v2_interacting"].fillna(0).astype(int).to_numpy()
  return {
    "frame_prec": float(precision_score(y_true, y_pred, zero_division=0)),
    "frame_rec": float(recall_score(y_true, y_pred, zero_division=0)),
    "frame_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "n_rows": int(len(y_true)),
    "n_positive_true": int(y_true.sum()),
    "n_positive_pred": int(y_pred.sum()),
  }


def _frame_metrics_merged(student: pd.DataFrame, feat: pd.DataFrame) -> dict:
  """Frame metrics after L∪R merge of both prediction and teacher labels."""
  feat_long = feat[["timestamp_ns", "t_sec", "hand_side", "object_name", "y_interacting"]].copy()
  feat_long["v2_interacting"] = feat_long["y_interacting"]
  student_m = merge_hands_pair_df(student)
  feat_m = merge_hands_pair_df(feat_long)
  return _frame_metrics_from_pairs(
    student_m.assign(hand_side="merged"),
    feat_m.rename(columns={"v2_interacting": "y_interacting"}).assign(hand_side="merged"),
  )


def eval_sequence(
  model,
  vocab,
  obj_to_id,
  device,
  seq_id: str,
  features_root: Path,
  teacher_root: Path,
  out_dir: Path,
  *,
  iou_threshold: float,
  merged_eval: bool = False,
  decode_margin: float = 0.0,
  decode_min_prob: float = 0.0,
  student_pairs: pd.DataFrame | None = None,
) -> dict:
  feat_path = features_root / seq_id / "features.csv"
  teacher_path = teacher_root / seq_id / "pair_scores_clean.csv"
  feat = pd.read_csv(feat_path)
  teacher = pd.read_csv(teacher_path)

  if student_pairs is not None:
    student = student_pairs.copy()
  else:
    student = pair_scores_for_sequence(
      seq_id, features_root, model, vocab, obj_to_id, device,
      decode_margin=decode_margin,
      decode_min_prob=decode_min_prob,
    )

  subdir = "prove"
  seq_out = out_dir / subdir / seq_id
  seq_out.mkdir(parents=True, exist_ok=True)
  student.to_csv(seq_out / "pair_scores_student.csv", index=False)

  if merged_eval:
    pair_m = eval_vs_teacher(student, teacher, iou_threshold=iou_threshold, merge_hands=True)
    frame_m = {k: pair_m[k] for k in ("frame_prec", "frame_rec", "frame_f1", "n_rows", "n_positive_true", "n_positive_pred")}
    tl_keys = ("seg_f1", "seg_prec", "seg_rec", "n_pred", "n_matched", "edit_sim")
    tl_flat = {k: pair_m[k] for k in tl_keys}
  else:
    teacher_ref = teacher[
      ["timestamp_ns", "t_sec", "hand_side", "object_name", "v2_interacting"]
    ].copy()
    pred_segs = pred_segments_from_pair_scores(student)
    gt_segs = pred_segments_from_pair_scores(teacher_ref)
    tl = evaluate_timeline(gt_segs, pred_segs, iou_threshold=iou_threshold)
    frame_m = _frame_metrics_from_pairs(student, feat)
    tl_flat = {
      "seg_f1": tl["seg_f1"],
      "seg_prec": tl["seg_prec"],
      "seg_rec": tl["seg_rec"],
      "n_pred": int(tl.get("overall", {}).get("n_pred", 0)),
      "n_matched": int(tl.get("overall", {}).get("n_matched", 0)),
      "edit_sim": tl["edit_sim"],
    }

  metrics = {
    "seq": seq_id,
    "merge_hands": bool(merged_eval),
    **frame_m,
    **tl_flat,
  }
  (seq_out / "summary.json").write_text(json.dumps(metrics, indent=2))
  return metrics


def main() -> None:
  args = parse_args()
  seqs = load_seq_list(args.seq_list)
  args.output_dir.mkdir(parents=True, exist_ok=True)

  model = vocab = obj_to_id = hparams = None
  device = None
  reuse_pairs = args.student_pairs_root is not None
  if not reuse_pairs:
    if not args.model.is_file():
      raise FileNotFoundError(args.model)
    if args.device == "auto":
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      device = torch.device(args.device)
    model, vocab, obj_to_id, hparams = load_gru_checkpoint(args.model, device)

  if args.merge_hands:
    merged_eval = True
  elif hparams is not None:
    merged_eval = hparams.get("label_mode") == "multilabel" or hparams.get("stream_target") == "merged"
  else:
    merged_eval = False

  per_seq = []
  for seq_id in seqs:
    print(f"eval {seq_id}", flush=True)
    student_pairs = None
    if reuse_pairs:
      csv_path = args.student_pairs_root / "prove" / seq_id / "pair_scores_student.csv"
      if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
      student_pairs = pd.read_csv(csv_path)
    m = eval_sequence(
      model,
      vocab,
      obj_to_id,
      device,
      seq_id,
      args.features_root,
      args.teacher_root,
      args.output_dir,
      iou_threshold=args.iou_threshold,
      merged_eval=merged_eval,
      decode_margin=args.decode_margin,
      decode_min_prob=args.decode_min_prob,
      student_pairs=student_pairs,
    )
    per_seq.append(m)
    print(
      f"  frame_f1={m['frame_f1']:.3f}  seg_f1={m['seg_f1']:.3f}  edit_sim={m['edit_sim']:.3f}",
      flush=True,
    )

  def _mean(key: str) -> float:
    vals = [float(m[key]) for m in per_seq]
    return float(sum(vals) / len(vals)) if vals else float("nan")

  n_pred_total = sum(int(m.get("n_pred", 0)) for m in per_seq)
  n_matched_total = sum(int(m.get("n_matched", 0)) for m in per_seq)
  mean_seg_prec = _mean("seg_prec")
  mean_seg_rec = _mean("seg_rec")
  spur_ratio = compute_spur_ratio(n_pred_total, n_matched_total)
  prove_f05 = compute_f05(mean_seg_prec, mean_seg_rec)
  penalized = prove_prec_penalized(mean_seg_prec, spur_ratio, penalty=args.prove_spur_penalty)
  mean_seg_f1 = _mean("seg_f1")
  mean_edit = _mean("edit_sim")
  combined = combine_prove_score(mean_seg_f1, mean_edit)

  summary = {
    "checkpoint": str(args.model),
    "n_seqs": len(per_seq),
    "iou_threshold": args.iou_threshold,
    "decode_margin": args.decode_margin,
    "decode_min_prob": args.decode_min_prob,
    "merge_hands": bool(merged_eval),
    "mean_frame_f1": _mean("frame_f1"),
    "mean_frame_prec": _mean("frame_prec"),
    "mean_frame_rec": _mean("frame_rec"),
    "mean_seg_f1": mean_seg_f1,
    "mean_seg_prec": mean_seg_prec,
    "mean_seg_rec": mean_seg_rec,
    "mean_edit_sim": mean_edit,
    "n_pred_total": n_pred_total,
    "n_matched_total": n_matched_total,
    "spur_ratio": spur_ratio,
    "prove_f05": prove_f05,
    "prove_prec_penalized": penalized,
    "combined_timeline_score": combined,
    "per_seq": [{k: v for k, v in m.items() if k != "timeline"} for m in per_seq],
  }
  out_path = args.output_dir / "prove_summary.json"
  out_path.write_text(json.dumps(summary, indent=2))
  print(json.dumps({k: v for k, v in summary.items() if k != "per_seq"}, indent=2))
  print(f"wrote {out_path}")


if __name__ == "__main__":
  main()
