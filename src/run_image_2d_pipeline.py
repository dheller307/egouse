#!/usr/bin/env python3
"""End-to-end image_2d pipeline: video → detectors → dual-dwell pair_scores_clean."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.paths import EGOUSE_ROOT, setup_paths

setup_paths()
from lib.pipelines.detectors import run_detector_pipeline
from lib.pipelines.image_2d_infer import run_image_2d_for_sequence
from lib.pipelines.pipeline_viz import plot_student_vs_teacher, write_eval_summary

DEFAULT_PREDS_ROOT = EGOUSE_ROOT / "outputs" / "student" / "features_raw"
DEFAULT_TEACHER_ROOT = EGOUSE_ROOT / "outputs" / "teacher"
DEFAULT_OUTPUT_ROOT = EGOUSE_ROOT / "outputs" / "baseline" / "image_2d"


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="image_2d pipeline: detector boxes → y_clean")
  src = p.add_mutually_exclusive_group(required=True)
  src.add_argument("--sequence", type=str, help="HOT3D sequence name")
  src.add_argument("--video-path", type=Path, help="Arbitrary MP4 (no teacher eval)")
  p.add_argument("--objects", type=str, default="", help="Comma-separated object names (--video-path)")
  p.add_argument("--output-dir", type=Path, default=None)
  p.add_argument("--preds-root", type=Path, default=None, help="Reuse existing preds (with --skip-detect)")
  p.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
  p.add_argument("--skip-detect", action="store_true")
  p.add_argument("--stride", type=int, default=1)
  p.add_argument("--no-cleanup", action="store_true")
  p.add_argument("--merge-hands", action="store_true")
  p.add_argument("--eval-vs-teacher", action="store_true", default=True)
  p.add_argument("--plot", action="store_true")
  return p.parse_args()


def _parse_objects(s: str) -> list[str]:
  return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
  args = parse_args()
  run_id = args.sequence or args.video_path.stem
  out_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / run_id)
  detect_dir = out_dir / "detect"
  object_names = _parse_objects(args.objects)

  if args.skip_detect and args.preds_root:
    preds_root = args.preds_root
  else:
    if args.video_path and not object_names:
      raise SystemExit("--objects required for --video-path")
    det = run_detector_pipeline(
      sequence=args.sequence,
      video_path=args.video_path,
      output_dir=detect_dir,
      stride=args.stride,
      object_names=object_names or None,
      skip_detect=args.skip_detect,
    )
    object_names = object_names or det.object_names
    preds_root = detect_dir

  if args.sequence:
    metrics = run_image_2d_for_sequence(
      args.sequence,
      preds_root=preds_root,
      teacher_root=args.teacher_root,
      output_dir=out_dir,
      no_cleanup=args.no_cleanup,
      merge_hands=args.merge_hands,
    )
    if args.eval_vs_teacher:
      write_eval_summary(out_dir / args.sequence / "summary_vs_teacher.json", metrics)
      print(
        f"vs teacher: frame_f1={metrics['frame_f1']:.3f} seg_f1={metrics['seg_f1']:.3f}",
        flush=True,
      )
    clean_path = out_dir / args.sequence / "pair_scores_clean.csv"
  else:
    metrics = run_image_2d_for_sequence(
      run_id,
      preds_root=preds_root,
      teacher_root=args.teacher_root,
      output_dir=out_dir,
      object_names=object_names,
      no_cleanup=args.no_cleanup,
      merge_hands=args.merge_hands,
    )
    clean_path = out_dir / "pair_scores_clean.csv"

  print(f"Wrote {clean_path}", flush=True)

  if args.plot and args.sequence:
    # Reuse student plot script with image_2d pair scores copied to infer layout if needed
    print(
      "image_2d plot: use plot_student_teacher_timeline with exported pair_scores_clean",
      flush=True,
    )


if __name__ == "__main__":
  main()
