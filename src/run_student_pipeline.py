#!/usr/bin/env python3
"""End-to-end student pipeline: video → detectors → features → GRU pair_scores.

Detector boxes are used raw (100DOH + YOLOE, ``--no-kalman``). No GT poses/boxes
enter features. Public eval/plots merge L∪R after inference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.paths import DEFAULT_STUDENT_CKPT, EGOUSE_ROOT, setup_paths

setup_paths()

DEFAULT_FEATURES_ROOT = EGOUSE_ROOT / "outputs" / "student" / "features_raw"
DEFAULT_TEACHER_ROOT = EGOUSE_ROOT / "outputs" / "teacher"
DEFAULT_INFER_ROOT = EGOUSE_ROOT / "outputs" / "student" / "infer"


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Student GRU pipeline: video → pair_scores.csv")
  src = p.add_mutually_exclusive_group(required=True)
  src.add_argument("--sequence", type=str, help="HOT3D sequence name")
  src.add_argument("--video-path", type=Path, help="Arbitrary MP4 (deploy; no teacher eval)")
  p.add_argument("--objects", type=str, default="", help="Comma-separated object names (--video-path)")
  p.add_argument("--checkpoint", type=Path, default=DEFAULT_STUDENT_CKPT)
  p.add_argument("--output-dir", type=Path, default=None)
  p.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
  p.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
  p.add_argument("--skip-detect", action="store_true")
  p.add_argument(
    "--use-features-raw",
    action="store_true",
    help="With --skip-detect, read preds from features-root/<seq>/ instead of output detect/",
  )
  p.add_argument("--stride", type=int, default=1)
  p.add_argument("--decode-margin", type=float, default=0.0)
  p.add_argument("--decode-min-prob", type=float, default=0.0)
  p.add_argument("--eval-vs-teacher", action="store_true")
  p.add_argument("--plot", action="store_true")
  p.add_argument("--device", type=str, default="auto")
  return p.parse_args()


def _parse_objects(s: str) -> list[str]:
  return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
  args = parse_args()

  import pandas as pd
  import torch

  from lib.pipelines.detectors import run_detector_pipeline
  from lib.pipelines.image_2d import load_preds_by_ts
  from lib.pipelines.image_2d_infer import eval_vs_teacher
  from lib.pipelines.pipeline_viz import plot_student_vs_teacher, write_eval_summary
  from lib.pipelines.student_infer import (
    build_features_deploy_grid,
    build_features_from_teacher_grid,
    infer_pair_scores,
    write_student_output,
  )

  device = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "auto"
    else torch.device(args.device)
  )

  run_id = args.sequence or args.video_path.stem
  out_dir = args.output_dir or (DEFAULT_INFER_ROOT / run_id)
  detect_dir = out_dir / "detect"
  object_names = _parse_objects(args.objects)
  if args.video_path and not object_names:
    raise SystemExit("--objects required for --video-path (comma-separated HOT3D-style names)")

  if args.sequence and args.skip_detect and not args.use_features_raw:
    # Reuse existing features.csv when only re-inferring
    feat_path = args.features_root / args.sequence / "features.csv"
    if feat_path.is_file():
      features_root = args.features_root
      infer_seq_id = args.sequence
      pair_scores = infer_pair_scores(
        infer_seq_id,
        features_root,
        args.checkpoint,
        device=device,
        decode_margin=args.decode_margin,
        decode_min_prob=args.decode_min_prob,
      )
      out_csv = write_student_output(
        pair_scores,
        out_dir,
        seq_id=run_id,
        checkpoint=args.checkpoint,
        extra_meta={"object_names": object_names},
      )
      print(f"Wrote {out_csv} ({len(pair_scores)} rows) [features cache]", flush=True)
      if args.eval_vs_teacher:
        teacher = pd.read_csv(args.teacher_root / args.sequence / "pair_scores_clean.csv")
        metrics = eval_vs_teacher(pair_scores, teacher, merge_hands=False)
        metrics["seq"] = args.sequence
        write_eval_summary(out_dir / "summary_vs_teacher.json", metrics)
        print(
          f"vs teacher: frame_f1={metrics['frame_f1']:.3f} seg_f1={metrics['seg_f1']:.3f}",
          flush=True,
        )
      if args.plot:
        png = plot_student_vs_teacher(
          args.sequence,
          checkpoint=args.checkpoint,
          features_root=features_root,
          teacher_root=args.teacher_root,
          output=out_dir / "student_vs_teacher.png",
        )
        print(f"Wrote {png}", flush=True)
      return

  if args.skip_detect and args.sequence and args.use_features_raw:
    detect_dir = args.features_root / args.sequence
    if not (detect_dir / "preds.jsonl").is_file():
      raise FileNotFoundError(f"Missing features_raw preds: {detect_dir / 'preds.jsonl'}")
    det = run_detector_pipeline(
      sequence=args.sequence,
      video_path=None,
      output_dir=detect_dir,
      skip_detect=True,
    )
  else:
    det = run_detector_pipeline(
      sequence=args.sequence,
      video_path=args.video_path,
      output_dir=detect_dir,
      stride=args.stride,
      object_names=object_names or None,
      skip_detect=args.skip_detect,
    )
  object_names = object_names or det.object_names
  preds_by_ts = load_preds_by_ts(det.preds_path)

  if args.sequence:
    teacher_clean = args.teacher_root / args.sequence / "pair_scores_clean.csv"
    if not teacher_clean.is_file():
      raise FileNotFoundError(f"Missing teacher labels: {teacher_clean}")
    features = build_features_from_teacher_grid(preds_by_ts, pd.read_csv(teacher_clean))
    features_root = args.features_root
    feat_dir = features_root / args.sequence
    feat_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(feat_dir / "features.csv", index=False)
    for name in ("preds.jsonl", "per_object.csv", "summary.json"):
      src = detect_dir / name
      if src.is_file():
        shutil.copy2(src, feat_dir / name)
    infer_seq_id = args.sequence
  else:
    features = build_features_deploy_grid(preds_by_ts, object_names)
    feat_dir = out_dir / run_id
    feat_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(feat_dir / "features.csv", index=False)
    features_root = out_dir
    infer_seq_id = run_id

  meta = {
    "backend": "raw_hybrid",
    "n_rows": len(features),
    "pred_frames": len(preds_by_ts),
    "object_names": object_names,
  }
  (feat_dir / "features_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

  pair_scores = infer_pair_scores(
    infer_seq_id,
    features_root,
    args.checkpoint,
    device=device,
    decode_margin=args.decode_margin,
    decode_min_prob=args.decode_min_prob,
  )
  out_csv = write_student_output(
    pair_scores,
    out_dir,
    seq_id=run_id,
    checkpoint=args.checkpoint,
    extra_meta={"object_names": object_names},
  )
  print(f"Wrote {out_csv} ({len(pair_scores)} rows)", flush=True)

  if args.eval_vs_teacher:
    if not args.sequence:
      raise SystemExit("--eval-vs-teacher requires --sequence (HOT3D)")
    teacher = pd.read_csv(args.teacher_root / args.sequence / "pair_scores_clean.csv")
    metrics = eval_vs_teacher(pair_scores, teacher, merge_hands=False)
    metrics["seq"] = args.sequence
    summary = write_eval_summary(out_dir / "summary_vs_teacher.json", metrics)
    print(
      f"vs teacher: frame_f1={metrics['frame_f1']:.3f} seg_f1={metrics['seg_f1']:.3f}",
      flush=True,
    )
    print(f"Wrote {summary}", flush=True)

  if args.plot and args.sequence:
    png = plot_student_vs_teacher(
      args.sequence,
      checkpoint=args.checkpoint,
      features_root=features_root,
      teacher_root=args.teacher_root,
      output=out_dir / "student_vs_teacher.png",
    )
    print(f"Wrote {png}", flush=True)


if __name__ == "__main__":
  main()
