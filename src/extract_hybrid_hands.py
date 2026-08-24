"""Extract 100DOH hand detections for hybrid pipeline (handobj venv)."""

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
  parser = argparse.ArgumentParser(description="Extract 100DOH hands for hybrid eval")
  parser.add_argument("--eval-dir", type=Path, required=True)
  parser.add_argument("--checkpoint", type=int, default=132028)
  parser.add_argument("--thresh-hand", type=float, default=0.5)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  frames_dir = args.eval_dir / "frames"
  gt_path = args.eval_dir / "gt.jsonl"
  if not gt_path.is_file():
    raise FileNotFoundError(gt_path)

  from lib.detectors.doh100 import Doh100Backend

  tag = "100doh_ego" if args.checkpoint == 132028 else "100doh_100k"
  backend = Doh100Backend(
    checkpoint=args.checkpoint,
    backend_tag=tag,
    thresh_hand=args.thresh_hand,
    thresh_obj=1.0,  # suppress obj inference use
  )

  out_path = args.eval_dir / "hands_preds.jsonl"
  n = 0
  with gt_path.open() as gf, out_path.open("w") as pf:
    for line in gf:
      gt_rec = json.loads(line)
      frame_path = frames_dir / gt_rec["frame"]
      image = cv2.imread(str(frame_path))
      if image is None:
        raise RuntimeError(f"Failed to read {frame_path}")
      dets = backend.infer(image)
      hands = [
        {
          "box": list(d.box),
          "score": d.score,
          "side": d.side,
          "contact": d.contact,
          "offset": list(d.offset) if d.offset else None,
        }
        for d in dets
        if d.label == "hand"
      ]
      pf.write(
        json.dumps(
          {
            "frame": gt_rec["frame"],
            "timestamp_ns": gt_rec["timestamp_ns"],
            "hands": hands,
          }
        )
        + "\n"
      )
      n += 1
      if n % 50 == 0:
        print(f"hands {n} frames")

  meta = {"backend": backend.backend_name, "n_frames": n, "checkpoint": args.checkpoint}
  (args.eval_dir / "hands_meta.json").write_text(json.dumps(meta, indent=2))
  print(f"wrote {out_path} ({n} frames)")


if __name__ == "__main__":
  main()
