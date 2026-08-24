#!/usr/bin/env python3
"""Prove-set image_2d from detector preds.jsonl (100DOH+YOLOE), eval vs teacher y*_clean.

Does NOT use HOT3D GT box2d. Boxes come from outputs/student/features_raw/<seq>/preds.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.paths import EGOUSE_ROOT, setup_paths

setup_paths()

from lib.experiments.teacher_pipeline import FROZEN_CLEANUP_CONFIG
from lib.pipelines.image_2d import load_preds_by_ts, score_sequence_from_preds
from lib.pipelines.image_2d_infer import densify_to_teacher_grid, eval_vs_teacher
from lib.scoring.dual_dwell_config import FROZEN_DUAL_DWELL
from lib.scoring.timeline_cleanup import cleanup_pair_scores
from lib.student_sequence_data import load_seq_list


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description="Prove image_2d (detector boxes) vs teacher y_clean"
  )
  p.add_argument(
    "--seq-list",
    type=Path,
    default=EGOUSE_ROOT / "config" / "seq_lists" / "prove.txt",
  )
  p.add_argument("--teacher-root", type=Path, default=EGOUSE_ROOT / "outputs" / "teacher")
  p.add_argument(
    "--preds-root",
    type=Path,
    default=EGOUSE_ROOT / "outputs" / "student" / "features_raw",
    help="Per-seq dir containing preds.jsonl (hybrid 100DOH+YOLOE)",
  )
  p.add_argument(
    "--output-dir",
    type=Path,
    default=EGOUSE_ROOT / "outputs" / "baseline" / "image_2d_det_prove",
  )
  p.add_argument("--iou-threshold", type=float, default=0.5)
  p.add_argument("--no-cleanup", action="store_true")
  p.add_argument("--merge-hands", action="store_true")
  p.add_argument(
    "--from-pair-scores-root",
    type=Path,
    default=None,
    help="Skip detector scoring; eval existing <seq>/pair_scores_clean.csv vs teacher.",
  )
  return p.parse_args()


def main() -> None:
  args = parse_args()
  seqs = load_seq_list(args.seq_list)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  config = FROZEN_DUAL_DWELL.to_scoring_config()
  per_seq = []

  for seq_id in seqs:
    print(f"image_2d(det) {seq_id}", flush=True)
    teacher_path = args.teacher_root / seq_id / "pair_scores_clean.csv"
    if not teacher_path.is_file():
      raise FileNotFoundError(teacher_path)
    teacher = pd.read_csv(teacher_path)
    seq_out = args.output_dir / seq_id
    seq_out.mkdir(parents=True, exist_ok=True)

    if args.from_pair_scores_root is not None:
      src_clean = args.from_pair_scores_root / seq_id / "pair_scores_clean.csv"
      if not src_clean.is_file():
        raise FileNotFoundError(src_clean)
      clean = pd.read_csv(src_clean)
      (seq_out / "box_source.json").write_text(
        json.dumps(
          {
            "box_source": "detector_preds",
            "reused_pair_scores": str(src_clean),
            "not_hot3d_gt_boxes": True,
          },
          indent=2,
        )
        + "\n"
      )
    else:
      preds_path = args.preds_root / seq_id / "preds.jsonl"
      if not preds_path.is_file():
        raise FileNotFoundError(preds_path)
      object_names = sorted({str(n) for n in teacher["object_name"].unique()})
      preds_by_ts = load_preds_by_ts(preds_path)
      teacher_ts = set(int(x) for x in teacher["timestamp_ns"].unique())
      preds_by_ts = {ts: rec for ts, rec in preds_by_ts.items() if ts in teacher_ts}

      raw_rows = score_sequence_from_preds(preds_by_ts, object_names, config)
      pair_df = pd.DataFrame(raw_rows)
      if pair_df.empty:
        raise RuntimeError(f"no pair rows for {seq_id}")

      t0 = int(teacher["timestamp_ns"].min())
      pair_df["t_sec"] = (pair_df["timestamp_ns"] - t0) / 1e9

      dense = densify_to_teacher_grid(pair_df, teacher)
      if args.no_cleanup:
        clean = dense
        cleanup_stats = {"skipped": True}
      else:
        clean, cleanup_stats = cleanup_pair_scores(dense, FROZEN_CLEANUP_CONFIG)

      pair_df.to_csv(seq_out / "pair_scores_raw_det.csv", index=False)
      dense.to_csv(seq_out / "pair_scores.csv", index=False)
      clean.to_csv(seq_out / "pair_scores_clean.csv", index=False)
      (seq_out / "cleanup_stats.json").write_text(json.dumps(cleanup_stats, indent=2) + "\n")
      (seq_out / "box_source.json").write_text(
        json.dumps(
          {
            "box_source": "detector_preds",
            "preds_path": str(preds_path),
            "backend_note": "100DOH hands + YOLOE objects (features_raw preds.jsonl)",
            "not_hot3d_gt_boxes": True,
          },
          indent=2,
        )
        + "\n"
      )

    m = eval_vs_teacher(
      clean,
      teacher,
      iou_threshold=args.iou_threshold,
      merge_hands=args.merge_hands,
    )
    m["seq"] = seq_id
    m["box_source"] = "detector_preds"
    (seq_out / "summary_vs_teacher.json").write_text(json.dumps(m, indent=2) + "\n")
    per_seq.append(m)
    print(
      f"  frame_f1={m['frame_f1']:.3f}  seg_f1={m['seg_f1']:.3f}  "
      f"seg_prec={m['seg_prec']:.3f}  fire={m['pair_fire_rate_pct']:.1f}%",
      flush=True,
    )

  def _mean(key: str) -> float:
    vals = [float(m[key]) for m in per_seq]
    return float(sum(vals) / len(vals)) if vals else float("nan")

  summary = {
    "method": "image_2d_detector",
    "box_source": "detector_preds",
    "not_hot3d_gt_boxes": True,
    "reference": "teacher_y_clean",
    "n_seqs": len(per_seq),
    "cleanup": not args.no_cleanup,
    "merge_hands": args.merge_hands,
    "iou_threshold": args.iou_threshold,
    "preset": FROZEN_DUAL_DWELL.name,
    "preds_root": str(args.preds_root),
    "mean_frame_prec": _mean("frame_prec"),
    "mean_frame_rec": _mean("frame_rec"),
    "mean_frame_f1": _mean("frame_f1"),
    "mean_seg_f1": _mean("seg_f1"),
    "mean_seg_prec": _mean("seg_prec"),
    "mean_seg_rec": _mean("seg_rec"),
    "mean_edit_sim": _mean("edit_sim"),
    "mean_pair_fire_rate_pct": _mean("pair_fire_rate_pct"),
    "per_seq": per_seq,
  }
  out = args.output_dir / "prove_summary_vs_teacher.json"
  out.write_text(json.dumps(summary, indent=2) + "\n")
  print(json.dumps({k: v for k, v in summary.items() if k != "per_seq"}, indent=2))
  print(f"Wrote {out}")


if __name__ == "__main__":
  main()
