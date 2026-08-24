#!/usr/bin/env python3
"""
Verify an EgoUse clone without HOT3D data or detector weights.

Checks documented entry points, unit tests, and core pipeline logic on toy fixtures.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

EGOUSE_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
README = EGOUSE_ROOT / "README.md"

# Paths that must exist after clone.
REQUIRED_PATHS = [
  "README.md",
  "requirements.txt",
  "config/seq_lists/regression.txt",
  "config/hot3d_yoloe_prompts.json",
  "checkpoints/phase_d_r7/best.pt",
  "src/run_teacher_pipeline.py",
  "src/run_student_pipeline.py",
  "src/run_image_2d_pipeline.py",
  "src/run_teacher_batch.py",
  "src/run_student_feature_batch.py",
  "src/regression_smoke.py",
  "src/aggregate_results.py",
  "scripts/regression_p0003.sh",
  "scripts/dump_video_frames.py",
  "lib/pipelines/detectors.py",
  "lib/pipelines/student_infer.py",
  "lib/pipelines/image_2d_infer.py",
  "docs/assets/demo.mp4",
]

CLI_HELP = [
  "src/run_teacher_pipeline.py",
  "src/run_student_pipeline.py",
  "src/run_image_2d_pipeline.py",
  "src/run_teacher_batch.py",
  "src/regression_smoke.py",
  "src/aggregate_results.py",
  "src/smoke_detector.py",
]

README_CMD_RE = re.compile(
  r"^\s*(?:python3?|bash)\s+((?:src|scripts)/[\w./_-]+\.(?:py|sh))\b",
  re.MULTILINE,
)


def _ok(msg: str) -> None:
  print(f"  OK {msg}")


def _fail(msg: str) -> None:
  print(f"  FAIL {msg}", file=sys.stderr)
  raise SystemExit(1)


def check_layout() -> None:
  print("== layout ==")
  for rel in REQUIRED_PATHS:
    path = EGOUSE_ROOT / rel
    if not path.is_file():
      _fail(f"missing {rel}")
    _ok(rel)


def check_readme_commands() -> None:
  print("== setup guide commands ==")
  if not README.is_file():
    _fail("README.md missing")
  text = README.read_text()
  cmds = sorted(set(README_CMD_RE.findall(text)))
  if not cmds:
    _ok("README has no src/scripts commands (optional)")
    return
  for rel in cmds:
    if not (EGOUSE_ROOT / rel).is_file():
      _fail(f"README references missing file: {rel}")
    _ok(f"documented: {rel}")


def check_cli_help() -> None:
  print("== CLI --help ==")
  py = sys.executable
  for rel in CLI_HELP:
    proc = subprocess.run(
      [py, str(EGOUSE_ROOT / rel), "--help"],
      cwd=EGOUSE_ROOT,
      capture_output=True,
      text=True,
    )
    if proc.returncode != 0:
      _fail(f"{rel} --help exited {proc.returncode}: {proc.stderr[:200]}")
    _ok(rel)


def run_unit_tests() -> None:
  print("== pytest ==")
  proc = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q"],
    cwd=EGOUSE_ROOT,
  )
  if proc.returncode != 0:
    _fail("pytest failed")
  _ok("tests/")


def simulate_pipelines() -> None:
  print("== pipeline simulation (fixtures) ==")
  sys.path.insert(0, str(EGOUSE_ROOT))
  from lib.paths import setup_paths

  setup_paths()

  import pandas as pd

  from lib.pipelines.image_2d import load_preds_by_ts
  from lib.pipelines.image_2d_infer import run_image_2d_for_sequence
  from lib.pipelines.student_infer import (
    build_features_deploy_grid,
    build_features_from_teacher_grid,
  )

  preds_path = FIXTURES / "preds.jsonl"
  teacher_path = FIXTURES / "teacher_pair_scores_clean.csv"
  if not preds_path.is_file() or not teacher_path.is_file():
    _fail("onboarding/fixtures incomplete")

  preds_by_ts = load_preds_by_ts(preds_path)
  teacher_df = pd.read_csv(teacher_path)

  feat_teacher = build_features_from_teacher_grid(preds_by_ts, teacher_df)
  if feat_teacher.empty or "hand_obj_iou" not in feat_teacher.columns:
    _fail("teacher-grid features empty")
  _ok(f"student features (teacher grid): {len(feat_teacher)} rows")

  feat_deploy = build_features_deploy_grid(preds_by_ts, ["mug_white", "keyboard"])
  if len(feat_deploy) < 6:
    _fail("deploy features too small")
  _ok(f"student features (deploy grid): {len(feat_deploy)} rows")

  with tempfile.TemporaryDirectory(prefix="egouse_onboard_") as tmp:
    tmp_path = Path(tmp)
    sim_preds = tmp_path / "sim"
    sim_preds.mkdir()
    sim_preds.joinpath("preds.jsonl").write_text(preds_path.read_text())
    teacher_root = tmp_path / "teacher"
    teacher_root.mkdir()
    seq = "sim_seq"
    (teacher_root / seq).mkdir()
    (teacher_root / seq / "pair_scores_clean.csv").write_text(teacher_path.read_text())

    metrics = run_image_2d_for_sequence(
      seq,
      preds_root=sim_preds,
      teacher_root=teacher_root,
      output_dir=tmp_path / "image_2d",
    )
    if "frame_f1" not in metrics:
      _fail("image_2d simulation missing metrics")
    clean = tmp_path / "image_2d" / seq / "pair_scores_clean.csv"
    if not clean.is_file():
      _fail("image_2d pair_scores_clean.csv not written")
    _ok(f"image_2d vs teacher: frame_f1={metrics['frame_f1']:.3f}")


def note_optional_local() -> None:
  print("== optional local assets (not required for this verify) ==")
  sys.path.insert(0, str(EGOUSE_ROOT))
  from lib.paths import resolve_student_checkpoint

  ckpt = resolve_student_checkpoint()
  for label, path in [
    ("GRU checkpoint", ckpt),
    ("HOT3D repo", EGOUSE_ROOT / "hot3d_repo"),
    ("HOT3D data", EGOUSE_ROOT / "hot3d_data"),
    ("100DOH clone", EGOUSE_ROOT / "third_party" / "Hand_Object_Detector"),
    ("venv_doh (100DOH)", EGOUSE_ROOT / "venv_doh"),
    ("venv_yoloe (YOLOE)", EGOUSE_ROOT / "venv_yoloe"),
  ]:
    status = "present" if path.exists() else "missing (optional; not in this snapshot)"
    print(f"  {label}: {status}")


def main() -> None:
  print(f"EgoUse onboarding verify ({EGOUSE_ROOT.name}/)\n")
  check_layout()
  check_readme_commands()
  check_cli_help()
  run_unit_tests()
  simulate_pipelines()
  note_optional_local()
  print("\nONBOARDING VERIFY PASS")
  print("Next: local HOT3D + detectors if you want full pipelines.")


if __name__ == "__main__":
  main()
