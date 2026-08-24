#!/usr/bin/env python3
"""Plot GRU student vs teacher y_clean swimlanes for one sequence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.paths import DEFAULT_STUDENT_CKPT, EGOUSE_ROOT
from lib.student_gru_infer import load_gru_checkpoint, pair_scores_for_sequence
from lib.viz.timeline import merge_hands_pair_df, plot_teacher_student_hoi_figure

DEFAULT_FEATURES_ROOT = EGOUSE_ROOT / "outputs" / "student" / "features_raw"
DEFAULT_TEACHER_ROOT = EGOUSE_ROOT / "outputs" / "teacher"
DEFAULT_CKPT = DEFAULT_STUDENT_CKPT


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="GRU student vs teacher timeline PNG")
  p.add_argument("--sequence", type=str, default="P0003_c701bd11")
  p.add_argument("--model", type=Path, default=DEFAULT_CKPT)
  p.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
  p.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
  p.add_argument("--output", type=Path, default=None)
  p.add_argument("--device", type=str, default="auto")
  p.add_argument("--merge-hands", action="store_true")
  p.add_argument("--tick-sec", type=float, default=10.0)
  p.add_argument("--t-min", type=float, default=None)
  p.add_argument("--t-max", type=float, default=None)
  p.add_argument("--object-fontsize", type=float, default=9.0)
  p.add_argument("--time-labelsize", type=float, default=8.0)
  p.add_argument("--figsize-w", type=float, default=None, help="Figure width in inches")
  p.add_argument("--figsize-h", type=float, default=None, help="Figure height in inches")
  p.add_argument("--bar-height", type=float, default=0.7)
  p.add_argument("--dash-linewidth", type=float, default=1.2)
  p.add_argument("--dpi", type=int, default=150)
  return p.parse_args()


def main() -> None:
  args = parse_args()
  seq = args.sequence
  feat_path = args.features_root / seq / "features.csv"
  teacher_path = args.teacher_root / seq / "pair_scores_clean.csv"
  if not feat_path.is_file():
    raise FileNotFoundError(feat_path)
  if not teacher_path.is_file():
    raise FileNotFoundError(teacher_path)
  if not args.model.is_file():
    raise FileNotFoundError(args.model)

  device = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "auto"
    else torch.device(args.device)
  )
  model, vocab, obj_to_id, hparams = load_gru_checkpoint(args.model, device)
  student = pair_scores_for_sequence(
    seq, args.features_root, model, vocab, obj_to_id, device
  )

  teacher = pd.read_csv(teacher_path)
  teacher_ref = teacher[
    ["timestamp_ns", "t_sec", "hand_side", "object_name", "v2_interacting"]
  ].copy()

  hands: tuple[str, ...] = ("left", "right")
  if args.merge_hands or hparams.get("label_mode") == "multilabel":
    student = merge_hands_pair_df(student)
    teacher_ref = merge_hands_pair_df(teacher_ref)
    hands = ("merged",)

  out = args.output or (
    EGOUSE_ROOT
    / "outputs"
    / "plots"
    / "student_review"
    / seq
    / "timeline_student_vs_teacher.png"
  )
  out.parent.mkdir(parents=True, exist_ok=True)

  figsize = None
  if args.figsize_w is not None and args.figsize_h is not None:
    figsize = (args.figsize_w, args.figsize_h)

  plot_teacher_student_hoi_figure(
    student,
    teacher_ref,
    tick_sec=args.tick_sec,
    output_path=out,
    title=f"{seq} — GRU student (fill) vs teacher y_clean (dashed)",
    hands=hands,
    fill_label="student",
    dashed_label="teacher",
    t_min=args.t_min,
    t_max=args.t_max,
    object_fontsize=args.object_fontsize,
    time_labelsize=args.time_labelsize,
    title_fontsize=max(args.object_fontsize + 2, 12),
    panel_title_fontsize=max(args.object_fontsize, 11),
    figsize=figsize,
    bar_height=args.bar_height,
    dash_linewidth=args.dash_linewidth,
    dpi=args.dpi,
  )
  student.to_csv(out.parent / "pair_scores_student.csv", index=False)
  print(f"Wrote {out}")


if __name__ == "__main__":
  main()
