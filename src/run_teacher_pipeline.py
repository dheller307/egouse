#!/usr/bin/env python3
"""End-to-end teacher pipeline: HOT3D VRS → y*_clean (+ optional plot)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.paths import EGOUSE_ROOT, setup_paths

setup_paths()

import pandas as pd

from lib.experiments.teacher_pipeline import ground_truth_paths_for_sequence
from lib.pipelines.pipeline_viz import plot_teacher_per_object, plot_teacher_vs_gt
from lib.pipelines.teacher_infer import PRESETS, resolve_sequence_path, run_teacher_for_sequence
from lib.viz.timeline import build_timeline_table, plot_per_object_hoi_figure


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Teacher pipeline: gt_world → y*_clean")
  p.add_argument("--sequence", type=str, required=True, help="HOT3D sequence name")
  p.add_argument("--output-dir", type=Path, default=None, help="Default: outputs/teacher/<SEQ>/")
  p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="frozen")
  p.add_argument("--skip-score", action="store_true")
  p.add_argument("--plot", action="store_true", help="Write timeline PNG(s)")
  p.add_argument("--plot-dir", type=Path, default=None)
  return p.parse_args()


def main() -> None:
  args = parse_args()
  out_dir = args.output_dir or (EGOUSE_ROOT / "outputs" / "teacher" / args.sequence)
  result = run_teacher_for_sequence(
    args.sequence,
    output_dir=out_dir,
    preset_key=args.preset,
    skip_score=args.skip_score,
  )
  print(
    f"Teacher {args.sequence}: {result.n_pair_rows} pairs, "
    f"{result.n_clean_on} clean ON, preset={result.preset_name}",
    flush=True,
  )
  if result.timeline_clean:
    tl = result.timeline_clean
    print(
      f"  timeline vs human GT: seg_f1={tl['seg_f1']:.3f} edit_sim={tl['edit_sim']:.3f}",
      flush=True,
    )
  print(f"Wrote {out_dir / 'pair_scores_clean.csv'}", flush=True)

  if not args.plot:
    return

  plot_dir = args.plot_dir or (EGOUSE_ROOT / "outputs" / "plots" / "pipeline" / args.sequence)
  plot_dir.mkdir(parents=True, exist_ok=True)
  if result.has_human_gt:
    plot_teacher_vs_gt(args.sequence, out_dir, output=plot_dir / "human_gt_vs_teacher.png")
    seq_path = resolve_sequence_path(args.sequence)
    gt_path, mapping_path, _ = ground_truth_paths_for_sequence(seq_path)
    timeline_df = build_timeline_table(
      out_dir / "frame_scores_clean.csv",
      gt_path,
      mapping_path,
    )
    clean = pd.read_csv(out_dir / "pair_scores_clean.csv")
    plot_per_object_hoi_figure(
      clean,
      timeline_df,
      tick_sec=10.0,
      output_path=plot_dir / "timeline_per_object.png",
    )
  else:
    plot_teacher_per_object(
      out_dir / "pair_scores_clean.csv",
      out_dir / "frame_scores_clean.csv",
      output_dir=plot_dir,
    )
  print(f"Wrote plots under {plot_dir}", flush=True)


if __name__ == "__main__":
  main()
