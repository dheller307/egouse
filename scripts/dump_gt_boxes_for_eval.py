#!/usr/bin/env python3
"""Dump RGB frames + rotated GT hand/object boxes for detector eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.paths import EGOUSE_ROOT, HOT3D_DATA_DIR, setup_paths

setup_paths()

from data_loaders.loader_hand_poses import LEFT_HAND_INDEX, RIGHT_HAND_INDEX
from lib.clip import resolve_clip_window, segments_path_for_sequence
from lib.geometry import bbox_to_ltrb
from lib.hot3d_io import RGB_STREAM_ID, get_hand_boxes, get_object_boxes, init_provider


def get_frame_bgr(timestamp_ns: int, device_data_provider):
  image_data = device_data_provider.get_image(timestamp_ns, RGB_STREAM_ID)
  image_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
  return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE), image_data.shape[0]


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Dump RGB + GT boxes for detector eval")
  parser.add_argument("--sequence", type=str, default="P0003_c701bd11")
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=EGOUSE_ROOT / "outputs" / "student" / "eval_100doh_p0003",
  )
  parser.add_argument("--stride", type=int, default=10, help="Stride within GT clip")
  parser.add_argument(
    "--visibility-threshold",
    type=float,
    default=0.1,
    help="Min visibility_ratio to keep a GT box",
  )
  parser.add_argument(
    "--jpeg-quality",
    type=int,
    default=85,
    help="JPEG quality for dumped frames (lower = less temp disk)",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_arguments()
  sequence_path = HOT3D_DATA_DIR / args.sequence
  object_library_path = HOT3D_DATA_DIR / "assets"
  segments_path = segments_path_for_sequence(sequence_path, EGOUSE_ROOT)
  mapping_path = sequence_path / "timecode_devicetime_mapping.csv"

  (
    _,
    device_data_provider,
    timestamps,
    hand_provider,
    object_provider,
    object_library,
  ) = init_provider(sequence_path, object_library_path)

  clip = resolve_clip_window(
    timestamps,
    start_frame=None,
    num_frames=None,
    stride=1,
    segments_path=segments_path,
    mapping_path=mapping_path,
  )
  time_slice = timestamps[
    clip.start_frame : clip.start_frame + clip.num_frames * clip.stride : clip.stride
  ]
  indices = list(range(0, len(time_slice), args.stride))

  frames_dir = args.output_dir / "frames"
  frames_dir.mkdir(parents=True, exist_ok=True)
  gt_path = args.output_dir / "gt.jsonl"

  n_written = 0
  with gt_path.open("w") as f:
    for out_idx, slice_idx in enumerate(indices):
      timestamp_ns = int(time_slice[slice_idx])
      frame, raw_h = get_frame_bgr(timestamp_ns, device_data_provider)
      frame_name = f"frame_{out_idx:04d}.jpg"
      cv2.imwrite(
        str(frames_dir / frame_name),
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
      )

      hands = []
      hand_result = get_hand_boxes(timestamp_ns, hand_provider)
      if hand_result is not None:
        for hand_uid, side in ((LEFT_HAND_INDEX, "left"), (RIGHT_HAND_INDEX, "right")):
          entry = hand_result["boxes"].get(hand_uid)
          if entry is None or entry.box2d is None:
            continue
          if entry.visibility_ratio < args.visibility_threshold:
            continue
          ltrb = bbox_to_ltrb(entry.box2d, raw_h, rotate=True)
          if ltrb is None:
            continue
          hands.append(
            {
              "side": side,
              "ltrb": [float(x) for x in ltrb],
              "vis": float(entry.visibility_ratio),
            }
          )

      objects = []
      object_result = get_object_boxes(timestamp_ns, object_provider)
      if object_result is not None:
        for object_uid, entry in object_result["boxes"].items():
          if entry.box2d is None or entry.visibility_ratio < args.visibility_threshold:
            continue
          ltrb = bbox_to_ltrb(entry.box2d, raw_h, rotate=True)
          if ltrb is None:
            continue
          name = object_library.object_id_to_name_dict[object_uid]
          objects.append(
            {
              "name": str(name),
              "uid": int(object_uid) if not isinstance(object_uid, str) else object_uid,
              "ltrb": [float(x) for x in ltrb],
              "vis": float(entry.visibility_ratio),
            }
          )

      rec = {
        "frame": frame_name,
        "frame_index": out_idx,
        "timestamp_ns": timestamp_ns,
        "hands": hands,
        "objects": objects,
      }
      f.write(json.dumps(rec) + "\n")
      n_written += 1
      if n_written % 50 == 0:
        print(f"wrote {n_written}/{len(indices)} frames")

  meta = {
    "sequence": args.sequence,
    "stride": args.stride,
    "n_frames": n_written,
    "clip_start_frame": clip.start_frame,
    "clip_num_frames": clip.num_frames,
    "visibility_threshold": args.visibility_threshold,
  }
  (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
  print(f"dumped {n_written} frames + gt.jsonl to {args.output_dir}")


if __name__ == "__main__":
  main()
