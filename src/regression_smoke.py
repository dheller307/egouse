#!/usr/bin/env python3
"""P0003 regression gate: pytest + teacher/student/image_2d metric parity vs frozen baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.paths import DEFAULT_STUDENT_CKPT, EGOUSE_ROOT, HOT3D_DATA_DIR

SEQ = "P0003_c701bd11"
TOL = 0.01

# Frozen regression targets (vs teacher y*_clean)
EXPECTED_GRU = {"frame_f1": 0.311, "seg_f1": 0.111}
EXPECTED_IMAGE2D = {"frame_f1": 0.393, "seg_f1": 0.350}

DEFAULT_CKPT = DEFAULT_STUDENT_CKPT
FEATURES_ROOT = EGOUSE_ROOT / "outputs" / "student" / "features_raw"
TEACHER_ROOT = EGOUSE_ROOT / "outputs" / "teacher"
IMAGE2D_ROOT = EGOUSE_ROOT / "outputs" / "baseline" / "image_2d_det_p0003"


def _check_metric(name: str, got: float, expected: float, tol: float) -> None:
  if abs(got - expected) > tol:
    raise SystemExit(
      f"REGRESSION FAIL {name}: got {got:.4f}, expected {expected:.4f} (tol={tol})"
    )
  print(f"  OK {name}: {got:.4f} (expected {expected:.4f})")


def run_pytest(egouse_root: Path) -> None:
  print("== pytest ==")
  subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q"],
    cwd=egouse_root,
    check=True,
  )


def check_teacher_artifacts(seq: str, *, regenerate: bool) -> None:
  print("== teacher artifacts ==")
  out = TEACHER_ROOT / seq
  clean = out / "pair_scores_clean.csv"
  if not clean.is_file():
    if regenerate:
      from lib.experiments.teacher_pipeline import FROZEN_CLEANUP_CONFIG, run_teacher_pipeline
      from lib.paths import setup_paths
      from lib.scoring.dual_dwell_config import FROZEN_DUAL_DWELL

      setup_paths()
      seq_path = HOT3D_DATA_DIR / seq
      if not seq_path.is_dir():
        raise SystemExit(f"Missing HOT3D sequence: {seq_path}")
      run_teacher_pipeline(
        seq_path,
        HOT3D_DATA_DIR / "assets",
        out,
        preset=FROZEN_DUAL_DWELL,
        cleanup_cfg=FROZEN_CLEANUP_CONFIG,
      )
    else:
      raise SystemExit(f"Missing teacher labels: {clean} (use --regenerate-teacher)")
  summary_path = out / "summary_teacher_timeline.json"
  if summary_path.is_file():
    summary = json.loads(summary_path.read_text())
    tl = summary.get("timeline_clean") or {}
    if tl:
      print(f"  teacher timeline_clean seg_f1={tl.get('seg_f1', float('nan')):.3f}")
  print(f"  OK teacher: {clean}")


def check_gru_metrics(seq: str, ckpt: Path) -> None:
  print("== GRU student vs teacher ==")
  if not ckpt.is_file():
    raise SystemExit(f"Missing checkpoint: {ckpt}")
  import torch

  from src.eval_student_gru import eval_sequence
  from lib.student_gru_infer import load_gru_checkpoint

  device = torch.device("cpu")
  model, vocab, obj_to_id, hparams = load_gru_checkpoint(ckpt, device)
  merged = hparams.get("label_mode") == "multilabel" or hparams.get("stream_target") == "merged"
  tmp_out = EGOUSE_ROOT / "outputs" / "student" / "_regression_smoke"
  m = eval_sequence(
    model,
    vocab,
    obj_to_id,
    device,
    seq,
    FEATURES_ROOT,
    TEACHER_ROOT,
    tmp_out,
    iou_threshold=0.5,
    merged_eval=merged,
  )
  _check_metric("gru.frame_f1", m["frame_f1"], EXPECTED_GRU["frame_f1"], TOL)
  _check_metric("gru.seg_f1", m["seg_f1"], EXPECTED_GRU["seg_f1"], TOL)


def check_image2d_metrics(seq: str) -> None:
  print("== image_2d (detector) vs teacher ==")
  summary_path = IMAGE2D_ROOT / seq / "summary_vs_teacher.json"
  if summary_path.is_file():
    m = json.loads(summary_path.read_text())
  else:
    from lib.pipelines.image_2d_infer import run_image_2d_for_sequence

    m = run_image_2d_for_sequence(
      seq,
      preds_root=FEATURES_ROOT,
      teacher_root=TEACHER_ROOT,
      output_dir=IMAGE2D_ROOT,
    )
    summary_path = IMAGE2D_ROOT / seq / "summary_vs_teacher.json"
  _check_metric("image2d.frame_f1", m["frame_f1"], EXPECTED_IMAGE2D["frame_f1"], TOL)
  _check_metric("image2d.seg_f1", m["seg_f1"], EXPECTED_IMAGE2D["seg_f1"], TOL)
  print(f"  OK summary: {summary_path}")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="P0003 regression smoke test")
  p.add_argument("--sequence", default=SEQ)
  p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
  p.add_argument("--tol", type=float, default=TOL)
  p.add_argument("--skip-pytest", action="store_true")
  p.add_argument("--regenerate-teacher", action="store_true")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  global TOL
  TOL = args.tol
  print(f"P0003 regression smoke ({args.sequence})", flush=True)
  if not args.skip_pytest:
    run_pytest(EGOUSE_ROOT)
  check_teacher_artifacts(args.sequence, regenerate=args.regenerate_teacher)
  check_gru_metrics(args.sequence, args.checkpoint)
  check_image2d_metrics(args.sequence)
  print("\nREGRESSION PASS", flush=True)


if __name__ == "__main__":
  main()
