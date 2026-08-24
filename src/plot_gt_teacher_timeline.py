#!/usr/bin/env python3
"""Plot human GT vs teacher y_clean swimlanes for P0003 regression (merged L∪R optional)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from lib.clip import mapping_path_for_sequence, segments_path_for_sequence
from lib.paths import EGOUSE_ROOT, HOT3D_DATA_DIR, setup_paths
from lib.viz.timeline import (
  build_timeline_table,
  gt_pair_df_from_timeline,
  merge_hands_pair_df,
  plot_teacher_student_hoi_figure,
)

setup_paths()

DEFAULT_SEQ = "P0003_c701bd11"
DEFAULT_TEACHER_ROOT = EGOUSE_ROOT / "outputs" / "teacher"


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Human GT vs teacher timeline PNG (P0003 regression)")
  p.add_argument(
    "--sequence",
    type=str,
    default=DEFAULT_SEQ,
    help="HOT3D sequence id (must have human GT CSV in src/data/ground_truth/)",
  )
  p.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
  p.add_argument(
    "--ground-truths-path",
    type=Path,
    default=None,
    help="Override GT segments CSV (default: from ground_truth_paths_for_sequence)",
  )
  p.add_argument(
    "--mapping-path",
    type=Path,
    default=None,
    help="Override timecode mapping CSV (default: hot3d_data/<seq>/...)",
  )
  p.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Default: outputs/plots/final/<SEQ>_human_gt_vs_teacher_merged.png",
  )
  p.add_argument(
    "--merge-hands",
    action="store_true",
    help="Single swimlane: union L/R ON per (t, object)",
  )
  p.add_argument("--tick-sec", type=float, default=10.0)
  p.add_argument("--t-min", type=float, default=None, help="Crop x-axis start (seconds)")
  p.add_argument("--t-max", type=float, default=None, help="Crop x-axis end (seconds)")
  p.add_argument("--object-fontsize", type=float, default=9.0)
  p.add_argument("--time-labelsize", type=float, default=8.0)
  return p.parse_args()


def main() -> None:
  args = parse_args()
  seq = args.sequence
  seq_path = HOT3D_DATA_DIR / seq
  teacher_path = args.teacher_root / seq / "pair_scores_clean.csv"
  frame_path = args.teacher_root / seq / "frame_scores_clean.csv"

  gt_path = args.ground_truths_path or segments_path_for_sequence(seq_path, EGOUSE_ROOT)
  mapping_path = args.mapping_path or mapping_path_for_sequence(seq_path)
  has_gt = gt_path.is_file()

  if not has_gt:
    raise SystemExit(
      f"No human GT for {seq} (expected {segments_path_for_sequence(seq_path, EGOUSE_ROOT)})"
    )
  if not teacher_path.is_file():
    raise FileNotFoundError(teacher_path)
  if not frame_path.is_file():
    raise FileNotFoundError(frame_path)
  if not mapping_path.is_file():
    raise FileNotFoundError(mapping_path)

  teacher = pd.read_csv(teacher_path)
  timeline_df = build_timeline_table(frame_path, gt_path, mapping_path)
  gt_pair = gt_pair_df_from_timeline(teacher, timeline_df)

  hands: tuple[str, ...] = ("left", "right")
  if args.merge_hands:
    gt_pair = merge_hands_pair_df(gt_pair)
    teacher = merge_hands_pair_df(teacher)
    hands = ("merged",)
    hand_note = "L∪R merged"
  else:
    hand_note = "per hand"

  title = f"{seq} — human GT (fill) vs teacher y_clean (dashed), {hand_note}"

  out = args.output
  if out is None:
    suffix = "_merged" if args.merge_hands else ""
    out = EGOUSE_ROOT / "outputs" / "plots" / "final" / f"{seq}_human_gt_vs_teacher{suffix}.png"

  out.parent.mkdir(parents=True, exist_ok=True)
  plot_teacher_student_hoi_figure(
    gt_pair,
    teacher,
    tick_sec=args.tick_sec,
    output_path=out,
    title=title,
    hands=hands,
    fill_label="human GT",
    dashed_label="teacher",
    t_min=args.t_min,
    t_max=args.t_max,
    object_fontsize=args.object_fontsize,
    time_labelsize=args.time_labelsize,
    title_fontsize=max(args.object_fontsize + 2, 12),
    panel_title_fontsize=max(args.object_fontsize, 11),
  )
  print(f"Wrote {out}")


if __name__ == "__main__":
  main()
