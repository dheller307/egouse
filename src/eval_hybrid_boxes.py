#!/usr/bin/env python3
"""Join 100DOH hands + YOLOE objects into preds.jsonl.

The student pipeline always runs with ``--no-kalman`` (raw detector boxes).
``KalmanBBoxTracker`` is an optional offline A/B, not used for training or
reported student metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.geometry import bbox_iou
from lib.paths import EGOUSE_ROOT
from lib.tracking.kalman_bbox import KalmanBBoxTracker


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Hybrid bbox eval vs HOT3D GT (raw or Kalman)")
  parser.add_argument(
    "--eval-dir",
    type=Path,
    default=EGOUSE_ROOT / "outputs" / "student" / "eval_hybrid_raw_p0003",
  )
  parser.add_argument("--iou-thresh", type=float, default=0.5)
  parser.add_argument("--yoloe-conf", type=float, default=0.25)
  parser.add_argument("--yoloe-size", default="s")
  parser.add_argument(
    "--no-kalman",
    action="store_true",
    help="Use raw YOLOE dets (student path). Default keeps offline Kalman A/B.",
  )
  parser.add_argument("--kalman-max-age", type=int, default=20)
  parser.add_argument("--kalman-iou", type=float, default=0.3)
  return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
  rows = []
  with path.open() as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def _match_hands(gt_hands: list[dict], pred_hands: list[dict], iou_thresh: float) -> list[dict]:
  used = set()
  rows = []
  for gt in gt_hands:
    gt_box = tuple(gt["ltrb"])
    side = gt["side"]
    candidates = [
      (i, p)
      for i, p in enumerate(pred_hands)
      if i not in used and (p.get("side") == side or p.get("side") is None)
    ]
    if not candidates:
      candidates = [(i, p) for i, p in enumerate(pred_hands) if i not in used]
    best_i, best_iou = None, 0.0
    for i, p in candidates:
      iou = float(bbox_iou(gt_box, tuple(p["box"])) or 0.0)
      if iou > best_iou:
        best_iou, best_i = iou, i
    if best_i is not None:
      used.add(best_i)
    rows.append({"side": side, "iou": best_iou, "tp": best_iou >= iou_thresh, "matched": best_i is not None})
  return rows


def _match_objects(gt_objs: list[dict], pred_objs: list[dict], iou_thresh: float) -> list[dict]:
  used = set()
  rows = []
  ordered = sorted(gt_objs, key=lambda g: -float(g.get("vis", 0)))
  for gt in ordered:
    gt_box = tuple(gt["ltrb"])
    best_i, best_iou = None, 0.0
    for i, p in enumerate(pred_objs):
      if i in used:
        continue
      iou = float(bbox_iou(gt_box, tuple(p["box"])) or 0.0)
      if iou > best_iou:
        best_iou, best_i = iou, i
    if best_i is not None and best_iou > 0:
      used.add(best_i)
    rows.append(
      {
        "name": gt["name"],
        "iou": best_iou,
        "tp": best_iou >= iou_thresh,
        "matched": best_i is not None and best_iou > 0,
      }
    )
  return rows


def main() -> None:
  args = parse_args()
  eval_dir = args.eval_dir
  frames_dir = eval_dir / "frames"
  gt_path = eval_dir / "gt.jsonl"
  hands_path = eval_dir / "hands_preds.jsonl"
  if not gt_path.is_file():
    raise FileNotFoundError(gt_path)
  if not hands_path.is_file():
    raise FileNotFoundError(f"Run extract_hybrid_hands.py first: {hands_path}")

  from lib.detectors.yoloe import YoloeBackend

  objs_backend = YoloeBackend(conf=args.yoloe_conf, yoloe_size=args.yoloe_size)
  tracker = None
  if not args.no_kalman:
    tracker = KalmanBBoxTracker(iou_thresh=args.kalman_iou, max_age=args.kalman_max_age)

  gt_rows = _load_jsonl(gt_path)
  hand_rows = _load_jsonl(hands_path)
  if len(gt_rows) != len(hand_rows):
    raise ValueError(f"gt/hands length mismatch: {len(gt_rows)} vs {len(hand_rows)}")

  preds_path = eval_dir / "preds.jsonl"
  hand_ious: list[float] = []
  hand_tps = hand_gts = 0
  obj_ious_by_name: dict[str, list[float]] = defaultdict(list)
  obj_tp_by_name: dict[str, int] = defaultdict(int)
  obj_gt_by_name: dict[str, int] = defaultdict(int)
  n_pred_hands = n_pred_objs = n_pred_hand_tp = n_pred_obj_tp = 0
  n_frames = 0
  n_coasting = 0

  with preds_path.open("w") as pf:
    for gt_rec, hand_rec in zip(gt_rows, hand_rows):
      if gt_rec["frame"] != hand_rec["frame"]:
        raise ValueError(f"frame mismatch {gt_rec['frame']} vs {hand_rec['frame']}")

      frame_path = frames_dir / gt_rec["frame"]
      image = cv2.imread(str(frame_path))
      if image is None:
        raise RuntimeError(f"Failed to read {frame_path}")

      pred_hands = hand_rec.get("hands") or []
      raw_objs = objs_backend.infer(image)
      det_dicts = [
        {"box": list(d.box), "score": d.score, "name": d.name}
        for d in raw_objs
        if d.label == "obj"
      ]
      if tracker is None:
        pred_objs = [
          {"box": list(d["box"]), "score": d["score"], "name": d["name"]} for d in det_dicts
        ]
      else:
        tracked = tracker.update(det_dicts)
        pred_objs = [
          {
            "box": list(t.box),
            "score": t.score,
            "name": t.name,
            "track_id": t.track_id,
            "coasting": t.coasting,
            "track_age": t.age,
          }
          for t in tracked
        ]
        n_coasting += sum(1 for t in tracked if t.coasting)

      n_pred_hands += len(pred_hands)
      n_pred_objs += len(pred_objs)

      hand_match_rows = _match_hands(gt_rec.get("hands", []), pred_hands, args.iou_thresh)
      obj_match_rows = _match_objects(gt_rec.get("objects", []), pred_objs, args.iou_thresh)

      for row in hand_match_rows:
        hand_gts += 1
        hand_ious.append(row["iou"])
        if row["tp"]:
          hand_tps += 1
          n_pred_hand_tp += 1

      used_pred = set()
      for gt in sorted(gt_rec.get("objects", []), key=lambda g: -float(g.get("vis", 0))):
        best_i, best_iou = None, 0.0
        for i, p in enumerate(pred_objs):
          if i in used_pred:
            continue
          iou = float(bbox_iou(tuple(gt["ltrb"]), tuple(p["box"])) or 0.0)
          if iou > best_iou:
            best_iou, best_i = iou, i
        if best_i is not None and best_iou >= args.iou_thresh:
          used_pred.add(best_i)
          n_pred_obj_tp += 1

      for row in obj_match_rows:
        name = row["name"]
        obj_gt_by_name[name] += 1
        obj_ious_by_name[name].append(row["iou"])
        if row["tp"]:
          obj_tp_by_name[name] += 1

      pf.write(
        json.dumps(
          {
            "frame": gt_rec["frame"],
            "timestamp_ns": gt_rec["timestamp_ns"],
            "hands": pred_hands,
            "objects": pred_objs,
            "hand_matches": hand_match_rows,
            "object_matches": obj_match_rows,
          }
        )
        + "\n"
      )
      n_frames += 1
      if n_frames % 50 == 0:
        print(f"scored {n_frames} frames")

  per_object = {}
  all_obj_ious: list[float] = []
  for name, ious in sorted(obj_ious_by_name.items()):
    all_obj_ious.extend(ious)
    n_gt = obj_gt_by_name[name]
    per_object[name] = {
      "n_gt": n_gt,
      "mean_iou": float(sum(ious) / len(ious)) if ious else 0.0,
      "recall_at_iou": float(obj_tp_by_name[name] / n_gt) if n_gt else 0.0,
    }

  tag = "raw" if args.no_kalman else "kalman"
  summary = {
    "backend": f"hybrid_100doh_hands_yoloe_26{args.yoloe_size}_{tag}",
    "yoloe_conf": args.yoloe_conf,
    "yoloe_size": args.yoloe_size,
    "no_kalman": bool(args.no_kalman),
    "n_frames": n_frames,
    "iou_thresh": args.iou_thresh,
    "n_coasting_tracks": 0 if args.no_kalman else n_coasting,
    "hands": {
      "n_gt": hand_gts,
      "mean_iou": float(sum(hand_ious) / len(hand_ious)) if hand_ious else 0.0,
      "recall_at_iou": float(hand_tps / hand_gts) if hand_gts else 0.0,
      "n_pred": n_pred_hands,
      "precision_at_iou": float(n_pred_hand_tp / n_pred_hands) if n_pred_hands else 0.0,
    },
    "objects": {
      "n_gt": sum(obj_gt_by_name.values()),
      "mean_iou": float(sum(all_obj_ious) / len(all_obj_ious)) if all_obj_ious else 0.0,
      "recall_at_iou": float(sum(obj_tp_by_name.values()) / sum(obj_gt_by_name.values()))
      if obj_gt_by_name
      else 0.0,
      "n_pred": n_pred_objs,
      "precision_at_iou": float(n_pred_obj_tp / n_pred_objs) if n_pred_objs else 0.0,
      "per_object": per_object,
    },
  }
  if not args.no_kalman:
    summary["kalman_max_age"] = args.kalman_max_age
    summary["kalman_iou"] = args.kalman_iou

  summary_path = eval_dir / "summary.json"
  summary_path.write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary, indent=2))

  csv_path = eval_dir / "per_object.csv"
  with csv_path.open("w") as cf:
    cf.write("object_name,n_gt,mean_iou,recall_at_0.5\n")
    for name, row in per_object.items():
      cf.write(f"{name},{row['n_gt']},{row['mean_iou']:.4f},{row['recall_at_iou']:.4f}\n")

  objs_backend.close()


if __name__ == "__main__":
  main()
