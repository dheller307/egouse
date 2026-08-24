"""Detector orchestration: HOT3D VRS or MP4 → preds.jsonl."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lib.paths import (
  EGOUSE_ROOT,
  HOT3D_DATA_DIR,
  VENV_DOH,
  VENV_MAIN,
  VENV_YOLOE,
  doh_torch_lib,
  venv_python,
)


@dataclass(frozen=True)
class DetectorRunResult:
  eval_dir: Path
  preds_path: Path
  n_frames: int
  object_names: list[str]


def _run(cmd: list[str], *, env: dict | None = None) -> None:
  print("+", " ".join(cmd), flush=True)
  subprocess.run(cmd, check=True, env=env)


def _load_preds_by_ts(path: Path) -> dict[int, dict]:
  by_ts: dict[int, dict] = {}
  with path.open() as f:
    for line in f:
      rec = json.loads(line)
      by_ts[int(rec["timestamp_ns"])] = rec
  return by_ts


def dump_hot3d_frames(
  sequence: str,
  output_dir: Path,
  *,
  stride: int = 1,
  jpeg_quality: int = 80,
) -> Path:
  venv_py = venv_python(VENV_MAIN)
  env = os.environ.copy()
  env.pop("LD_LIBRARY_PATH", None)
  _run(
    [
      str(venv_py),
      str(EGOUSE_ROOT / "scripts" / "dump_gt_boxes_for_eval.py"),
      "--sequence",
      sequence,
      "--stride",
      str(stride),
      "--output-dir",
      str(output_dir),
      "--jpeg-quality",
      str(jpeg_quality),
    ],
    env=env,
  )
  return output_dir


def dump_video_frames(
  video_path: Path,
  output_dir: Path,
  *,
  stride: int = 1,
  jpeg_quality: int = 80,
  max_frames: int | None = None,
) -> Path:
  venv_py = venv_python(VENV_MAIN)
  cmd = [
    str(venv_py),
    str(EGOUSE_ROOT / "scripts" / "dump_video_frames.py"),
    "--video-path",
    str(video_path),
    "--output-dir",
    str(output_dir),
    "--stride",
    str(stride),
    "--jpeg-quality",
    str(jpeg_quality),
  ]
  if max_frames is not None:
    cmd.extend(["--max-frames", str(max_frames)])
  _run(cmd)
  return output_dir


def run_hybrid_detectors(
  eval_dir: Path,
  *,
  yoloe_size: str = "s",
  no_kalman: bool = True,
) -> DetectorRunResult:
  """Run 100DOH hands + YOLOE objects. Default (and student path): raw boxes, no Kalman."""
  venv = venv_python(VENV_MAIN)
  venv_doh = venv_python(VENV_DOH)
  venv_yoloe = venv_python(VENV_YOLOE)

  env_clean = os.environ.copy()
  env_clean.pop("LD_LIBRARY_PATH", None)

  env_hand = env_clean.copy()
  torch_lib = doh_torch_lib()
  if torch_lib is not None:
    env_hand["LD_LIBRARY_PATH"] = str(torch_lib)

  _run(
    [str(venv_doh), str(EGOUSE_ROOT / "src" / "extract_hybrid_hands.py"), "--eval-dir", str(eval_dir)],
    env=env_hand,
  )

  hybrid_cmd = [
    str(venv_yoloe),
    str(EGOUSE_ROOT / "src" / "eval_hybrid_boxes.py"),
    "--eval-dir",
    str(eval_dir),
    "--yoloe-size",
    yoloe_size,
  ]
  if no_kalman:
    hybrid_cmd.append("--no-kalman")
  _run(hybrid_cmd, env=env_clean)

  preds_path = eval_dir / "preds.jsonl"
  if not preds_path.is_file():
    raise FileNotFoundError(preds_path)

  meta_path = eval_dir / "meta.json"
  n_frames = 0
  if meta_path.is_file():
    n_frames = int(json.loads(meta_path.read_text()).get("n_frames", 0))
  if not n_frames:
    n_frames = sum(1 for _ in preds_path.open())

  object_names: list[str] = []
  per_obj = eval_dir / "per_object.csv"
  if per_obj.is_file():
    import csv

    with per_obj.open() as f:
      reader = csv.DictReader(f)
      object_names = [row["object_name"] for row in reader if row.get("object_name")]
  if not object_names:
    preds = _load_preds_by_ts(preds_path)
    names: set[str] = set()
    for rec in preds.values():
      for obj in rec.get("objects") or []:
        if obj.get("name"):
          names.add(str(obj["name"]))
    object_names = sorted(names)

  return DetectorRunResult(
    eval_dir=eval_dir,
    preds_path=preds_path,
    n_frames=n_frames,
    object_names=object_names,
  )


def run_detector_pipeline(
  *,
  sequence: str | None = None,
  video_path: Path | None = None,
  output_dir: Path,
  tmp_root: Path | None = None,
  stride: int = 1,
  jpeg_quality: int = 80,
  yoloe_size: str = "s",
  object_names: list[str] | None = None,
  skip_detect: bool = False,
) -> DetectorRunResult:
  """
  Full detector path into output_dir (contains preds.jsonl).

  HOT3D: dumps frames to tmp then copies artifacts to output_dir.
  MP4: requires object_names for downstream feature grid (stored in meta).
  """
  if skip_detect:
    preds = output_dir / "preds.jsonl"
    if not preds.is_file():
      raise FileNotFoundError(f"--skip-detect but missing {preds}")
    meta = json.loads((output_dir / "meta.json").read_text()) if (output_dir / "meta.json").is_file() else {}
    names = object_names or meta.get("object_names") or []
    return DetectorRunResult(eval_dir=output_dir, preds_path=preds, n_frames=0, object_names=list(names))

  if bool(sequence) == bool(video_path):
    raise ValueError("Provide exactly one of sequence= or video_path=")

  output_dir.mkdir(parents=True, exist_ok=True)
  tmp = tmp_root or Path("/freespace/local/winlab/.tmp")
  tmp.mkdir(parents=True, exist_ok=True)
  work = tmp / f"detect_{os.getpid()}"
  if work.exists():
    shutil.rmtree(work)
  work.mkdir(parents=True)

  try:
    if sequence:
      dump_hot3d_frames(sequence, work, stride=stride, jpeg_quality=jpeg_quality)
    else:
      dump_video_frames(video_path, work, stride=stride, jpeg_quality=jpeg_quality)

    result = run_hybrid_detectors(work, yoloe_size=yoloe_size, no_kalman=True)

    for name in ("preds.jsonl", "summary.json", "hands_preds.jsonl", "hands_meta.json", "per_object.csv", "meta.json"):
      src = work / name
      if src.is_file():
        shutil.copy2(src, output_dir / name)

    names = object_names or result.object_names
    if object_names:
      meta = json.loads((output_dir / "meta.json").read_text()) if (output_dir / "meta.json").is_file() else {}
      meta["object_names"] = list(object_names)
      (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    return DetectorRunResult(
      eval_dir=output_dir,
      preds_path=output_dir / "preds.jsonl",
      n_frames=result.n_frames,
      object_names=list(names),
    )
  finally:
    if work.exists():
      shutil.rmtree(work, ignore_errors=True)
