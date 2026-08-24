#!/usr/bin/env python3
"""Dump RGB frames from an arbitrary video file (MP4 etc.) for detector pipeline."""

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
  p = argparse.ArgumentParser(description="Dump video frames + synthetic gt.jsonl for detectors")
  p.add_argument("--video-path", type=Path, required=True)
  p.add_argument("--output-dir", type=Path, required=True)
  p.add_argument("--stride", type=int, default=1, help="Frame stride (1 = every frame)")
  p.add_argument("--jpeg-quality", type=int, default=80)
  p.add_argument(
    "--fps-assumed",
    type=float,
    default=None,
    help="Override FPS for synthetic timestamp_ns (default: probe from video)",
  )
  p.add_argument("--max-frames", type=int, default=None)
  return p.parse_args()


def main() -> None:
  args = parse_args()
  if not args.video_path.is_file():
    raise SystemExit(f"Missing video: {args.video_path}")

  cap = cv2.VideoCapture(str(args.video_path))
  if not cap.isOpened():
    raise SystemExit(f"Failed to open video: {args.video_path}")

  fps = args.fps_assumed or float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
  if fps <= 0:
    fps = 30.0

  frames_dir = args.output_dir / "frames"
  frames_dir.mkdir(parents=True, exist_ok=True)
  gt_path = args.output_dir / "gt.jsonl"

  n_written = 0
  frame_idx = 0
  out_idx = 0
  with gt_path.open("w") as gf:
    while True:
      ok, frame_bgr = cap.read()
      if not ok:
        break
      if frame_idx % args.stride != 0:
        frame_idx += 1
        continue
      if args.max_frames is not None and n_written >= args.max_frames:
        break

      # Match HOT3D dump orientation (90° clockwise).
      frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
      frame_name = f"frame_{out_idx:04d}.jpg"
      cv2.imwrite(
        str(frames_dir / frame_name),
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
      )
      timestamp_ns = int(round((frame_idx / fps) * 1e9))
      rec = {
        "frame": frame_name,
        "frame_index": out_idx,
        "timestamp_ns": timestamp_ns,
        "hands": [],
        "objects": [],
      }
      gf.write(json.dumps(rec) + "\n")
      n_written += 1
      out_idx += 1
      frame_idx += 1
      if n_written % 100 == 0:
        print(f"wrote {n_written} frames", flush=True)

  cap.release()
  meta = {
    "source": "video",
    "video_path": str(args.video_path.resolve()),
    "stride": args.stride,
    "fps": fps,
    "n_frames": n_written,
  }
  (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
  print(f"dumped {n_written} frames to {args.output_dir}")


if __name__ == "__main__":
  main()
