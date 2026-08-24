#!/usr/bin/env python3
"""Smoke-test 100DOH or YOLOE on a folder of JPEG frames (setup verification)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.paths import EGOUSE_ROOT


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run 100DOH or YOLOE on dumped frames; writes smoke_summary.json"
  )
  parser.add_argument("--frames-dir", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument(
    "--backend",
    choices=["auto", "100doh", "yoloe"],
    default="auto",
    help="auto: try 100DOH, then YOLOE",
  )
  parser.add_argument("--checkpoint", type=int, default=132028, help="100DOH weights id")
  parser.add_argument("--yoloe-size", default="s")
  return parser.parse_args()


def _load_backend(name: str, checkpoint: int, *, yoloe_size: str):
  if name == "yoloe":
    from lib.detectors.yoloe import YoloeBackend

    return YoloeBackend(yoloe_size=yoloe_size)
  if name == "100doh":
    from lib.detectors.doh100 import Doh100Backend

    tag = "100doh_ego" if checkpoint == 132028 else "100doh_100k"
    return Doh100Backend(checkpoint=checkpoint, backend_tag=tag)
  raise ValueError(f"unknown backend {name}")


def _try_auto(checkpoint: int, *, yoloe_size: str):
  errors: list[str] = []
  for name in ("100doh", "yoloe"):
    try:
      return _load_backend(name, checkpoint, yoloe_size=yoloe_size), errors
    except Exception as exc:
      errors.append(f"{name}: {exc}")
  raise RuntimeError("no detector backend available:\n" + "\n".join(errors))


def _count_stats(detections) -> dict:
  n_hand = sum(1 for d in detections if d.label == "hand")
  n_obj = sum(1 for d in detections if d.label == "obj")
  return {"n_hand": n_hand, "n_obj": n_obj}


def main() -> None:
  args = parse_args()
  frames = sorted(args.frames_dir.glob("frame_*.jpg"))
  if not frames:
    raise FileNotFoundError(f"No frame_*.jpg in {args.frames_dir}")

  args.out_dir.mkdir(parents=True, exist_ok=True)
  overlay_dir = args.out_dir / "overlays"
  overlay_dir.mkdir(parents=True, exist_ok=True)

  fallback_errors: list[str] = []
  if args.backend == "auto":
    backend, fallback_errors = _try_auto(args.checkpoint, yoloe_size=args.yoloe_size)
  else:
    backend = _load_backend(args.backend, args.checkpoint, yoloe_size=args.yoloe_size)

  per_frame = []
  for frame_path in frames:
    image = cv2.imread(str(frame_path))
    if image is None:
      raise RuntimeError(f"Failed to read {frame_path}")
    dets = backend.infer(image)
    stats = _count_stats(dets)
    stats["frame"] = frame_path.name
    per_frame.append(stats)

  n_hand_frames = sum(1 for s in per_frame if s["n_hand"] >= 1)
  n_obj_frames = sum(1 for s in per_frame if s["n_obj"] >= 1)
  need_hands = backend.backend_name.startswith("100doh")
  passed = (n_hand_frames >= 1) if need_hands else (n_obj_frames >= 1)

  summary = {
    "backend": backend.backend_name,
    "cuda": getattr(backend, "cuda", None),
    "n_frames": len(frames),
    "n_hand_frames": n_hand_frames,
    "n_obj_frames": n_obj_frames,
    "pass": passed,
    "per_frame": per_frame,
    "fallback_errors": fallback_errors or None,
  }
  summary_path = args.out_dir / "smoke_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2) + "\n")
  print(json.dumps(summary, indent=2))

  if hasattr(backend, "close"):
    backend.close()

  if not passed:
    raise SystemExit(
      f"Smoke failed: backend={backend.backend_name} "
      f"hands={n_hand_frames}/{len(frames)} objs={n_obj_frames}/{len(frames)}"
    )


if __name__ == "__main__":
  main()
