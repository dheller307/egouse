"""Shared plot/eval helpers for end-to-end pipeline CLIs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from lib.paths import EGOUSE_ROOT


def run_subprocess(cmd: list[str]) -> None:
  print("+", " ".join(cmd), flush=True)
  subprocess.run(cmd, check=True)


def plot_teacher_vs_gt(
  sequence: str,
  teacher_root: Path,
  *,
  output: Path | None = None,
  merge_hands: bool = True,
) -> Path:
  out = output or (
    EGOUSE_ROOT / "outputs" / "plots" / "pipeline" / sequence / "teacher_vs_gt.png"
  )
  cmd = [
    sys.executable,
    str(EGOUSE_ROOT / "src" / "plot_gt_teacher_timeline.py"),
    "--sequence",
    sequence,
    "--teacher-root",
    str(teacher_root),
    "--output",
    str(out),
  ]
  if merge_hands:
    cmd.append("--merge-hands")
  run_subprocess(cmd)
  return out


def plot_student_vs_teacher(
  sequence: str,
  *,
  checkpoint: Path,
  features_root: Path,
  teacher_root: Path,
  output: Path | None = None,
  merge_hands: bool = True,
) -> Path:
  out = output or (
    EGOUSE_ROOT / "outputs" / "plots" / "pipeline" / sequence / "student_vs_teacher.png"
  )
  cmd = [
    sys.executable,
    str(EGOUSE_ROOT / "src" / "plot_student_teacher_timeline.py"),
    "--sequence",
    sequence,
    "--backend",
    "gru",
    "--model",
    str(checkpoint),
    "--features-root",
    str(features_root),
    "--teacher-root",
    str(teacher_root),
    "--output",
    str(out),
  ]
  if merge_hands:
    cmd.append("--merge-hands")
  run_subprocess(cmd)
  return out


def plot_teacher_per_object(
  pair_scores_clean: Path,
  frame_scores_clean: Path,
  *,
  output_dir: Path,
) -> Path:
  cmd = [
    sys.executable,
    str(EGOUSE_ROOT / "src" / "plot_timeline.py"),
    "--frame_scores_path",
    str(frame_scores_clean),
    "--pair_scores_path",
    str(pair_scores_clean),
    "--rule",
    "per_object",
    "--output_dir",
    str(output_dir),
  ]
  run_subprocess(cmd)
  return output_dir / "timeline_per_object.png"


def write_eval_summary(path: Path, metrics: dict) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(metrics, indent=2) + "\n")
  return path
