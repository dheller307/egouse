"""image_2d dual-dwell from detector boxes (100DOH hands + YOLOE objects).

Does NOT use HOT3D GT box2d providers — only preds.jsonl from the hybrid detector
pipeline (same boxes the student features are built from).
"""

from __future__ import annotations

import json
from pathlib import Path

from lib.geometry import bbox_iou, distance_confidence, norm_center_distance, spatial_confidence
from lib.scoring.co_motion import co_motion_ok, co_motion_score
from lib.scoring.interaction_v2 import V2DwellTracker
from lib.scoring.motion import HandDynamicHistory, ObjectDynamicHistory
from lib.scoring.pipeline import ScoringConfig, v2_proximity_ok


def load_preds_by_ts(path: Path) -> dict[int, dict]:
  by_ts: dict[int, dict] = {}
  with path.open() as f:
    for line in f:
      rec = json.loads(line)
      for obj in rec.get("objects") or []:
        if obj.get("coasting"):
          raise ValueError(
            f"coasting object in preds at ts={rec.get('timestamp_ns')} — refuse Kalman preds"
          )
      by_ts[int(rec["timestamp_ns"])] = rec
  return by_ts


def best_hand_for_side(hands: list[dict], side: str) -> dict | None:
  """Match student feature builder: prefer labeled side, else best hand overall."""
  sided = [h for h in hands if h.get("side") == side]
  if sided:
    return max(sided, key=lambda h: float(h.get("score") or 0.0))
  return max(hands, key=lambda h: float(h.get("score") or 0.0)) if hands else None


def best_obj_for_name(objects: list[dict], object_name: str) -> dict | None:
  named = [o for o in objects if o.get("name") == object_name]
  if named:
    return max(named, key=lambda o: float(o.get("score") or 0.0))
  return None


def _ltrb(box) -> tuple[float, float, float, float] | None:
  if box is None or len(box) < 4:
    return None
  return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def _score_pair_metrics(hand_ltrb, object_ltrb, config: ScoringConfig) -> dict:
  iou = bbox_iou(hand_ltrb, object_ltrb)
  dist_score = distance_confidence(hand_ltrb, object_ltrb, config.k)
  return {
    "iou": float(iou or 0.0),
    "d_norm": float(norm_center_distance(hand_ltrb, object_ltrb) or 0.0),
    "dist_conf": float(dist_score or 0.0),
    "spatial_conf": float(spatial_confidence(iou or 0.0, dist_score or 0.0)),
  }


def score_sequence_from_preds(
  preds_by_ts: dict[int, dict],
  object_names: list[str],
  config: ScoringConfig,
  *,
  min_hand_score: float = 0.0,
  min_obj_score: float = 0.0,
) -> list[dict]:
  """
  Run image-plane dual-dwell on detector boxes.

  Timestamps follow preds_by_ts (sorted). Pair grid = every (hand, object_name)
  in ``object_names`` each frame (missing dets → OFF / decay).
  """
  name_to_uid = {n: i + 1 for i, n in enumerate(object_names)}
  hand_hist = HandDynamicHistory(
    max_history_len=config.max_history_len,
    smooth_frames=config.smooth_frames,
    min_var_frames=config.min_var_frames,
    gap_reset_ms=config.gap_reset_ms,
  )
  obj_hist = ObjectDynamicHistory(
    max_history_len=config.max_history_len,
    smooth_frames=config.smooth_frames,
    min_var_frames=config.min_var_frames,
    gap_reset_ms=config.gap_reset_ms,
  )
  tracker = V2DwellTracker(
    iou_dwell_ms=config.v2_iou_dwell_ms,
    cm_dwell_ms=config.v2_co_motion_dwell_ms,
    iou_gap_tolerance_ms=config.v2_iou_gap_tolerance_ms,
    cm_gap_tolerance_ms=config.v2_co_motion_gap_tolerance_ms,
  )

  timestamps = sorted(preds_by_ts.keys())
  if not timestamps:
    return []
  t0 = timestamps[0]
  pair_rows: list[dict] = []
  prev_ts: int | None = None

  for frame_index, ts in enumerate(timestamps):
    dt_ns = 33_300_000 if prev_ts is None else int(ts - prev_ts)
    prev_ts = ts
    t_sec = (ts - t0) / 1e9
    rec = preds_by_ts[ts]
    hands = rec.get("hands") or []
    objects = rec.get("objects") or []

    # Update each visible object once per frame (shared across hands).
    obj_motion_by_uid: dict[int, object] = {}
    obj_ltrb_by_uid: dict[int, tuple[float, float, float, float]] = {}
    obj_meta_by_uid: dict[int, dict] = {}
    visible_obj_uids: set[int] = set()
    for object_name in object_names:
      obj = best_obj_for_name(objects, object_name)
      if obj is None or float(obj.get("score") or 0.0) < min_obj_score:
        continue
      oltrb = _ltrb(obj.get("box"))
      if oltrb is None:
        continue
      uid = name_to_uid[object_name]
      visible_obj_uids.add(uid)
      obj_ltrb_by_uid[uid] = oltrb
      obj_meta_by_uid[uid] = obj
      obj_motion_by_uid[uid] = obj_hist.update(uid, oltrb, dt_ns, ts)

    obj_hist.decay_offscreen(visible_obj_uids)

    for hand_side in ("left", "right"):
      hand = best_hand_for_side(hands, hand_side)
      if hand is None or float(hand.get("score") or 0.0) < min_hand_score:
        tracker.decay_hand(hand_side, dt_ns)
        continue
      hltrb = _ltrb(hand.get("box"))
      if hltrb is None:
        tracker.decay_hand(hand_side, dt_ns)
        continue

      hand_motion = hand_hist.update(hand_side, hltrb, dt_ns, ts)
      iou_ok_by_object: dict[int, bool] = {}
      cm_ok_by_object: dict[int, bool] = {}
      pending: list[dict] = []

      for object_name in object_names:
        uid = name_to_uid[object_name]
        if uid not in obj_ltrb_by_uid:
          continue
        oltrb = obj_ltrb_by_uid[uid]
        obj = obj_meta_by_uid[uid]
        metrics = _score_pair_metrics(hltrb, oltrb, config)
        prox = {
          "iou": metrics["iou"],
          "iou_interacting": metrics["iou"] > config.iou_threshold,
          "dist_conf": metrics["dist_conf"],
          "dist_interacting": metrics["dist_conf"] > config.dist_threshold,
          "spatial_conf": metrics["spatial_conf"],
          "spatial_interacting": metrics["spatial_conf"] > config.spatial_threshold,
        }
        iou_ok = v2_proximity_ok(prox)
        obj_motion = obj_motion_by_uid[uid]
        cm_score = co_motion_score(
          hand_motion.velocity,
          obj_motion.velocity,
          config.co_motion_v_min,
        )
        cm_ok = co_motion_ok(cm_score, config.co_motion_threshold)
        frame_ok = iou_ok and cm_ok
        iou_ok_by_object[uid] = iou_ok
        cm_ok_by_object[uid] = cm_ok
        pending.append({
          "object_uid": uid,
          "object_name": object_name,
          "iou": metrics["iou"],
          "co_motion_score": cm_score,
          "co_motion_ok": cm_ok,
          "iou_ok": iou_ok,
          "frame_ok": frame_ok,
          "obj_score": float(obj.get("score") or 0.0),
          "hand_score": float(hand.get("score") or 0.0),
        })

      if not pending:
        tracker.decay_hand(hand_side, dt_ns)
        continue

      v2_result = tracker.update_hand_frame(
        hand_side,
        iou_ok_by_object,
        cm_ok_by_object,
        dt_ns,
      )
      for row in pending:
        uid = row["object_uid"]
        pair_rows.append({
          "timestamp_ns": ts,
          "t_sec": t_sec,
          "frame_index": frame_index,
          "hand_side": hand_side,
          "object_uid": uid,
          "object_name": row["object_name"],
          "iou": row["iou"],
          "v2_aabb_dist_m": float("nan"),
          "v2_iou_ok": int(row["iou_ok"]),
          "v2_cm_prox_ok": int(row["iou_ok"]),
          "v2_co_motion_score": row["co_motion_score"],
          "v2_co_motion_ok": int(row["co_motion_ok"]),
          "v2_stationary_ok": 0,
          "v2_slow_manip_ok": 0,
          "v2_motion_ok": int(row["co_motion_ok"] and row["iou_ok"]),
          "v2_frame_ok": int(row["frame_ok"]),
          "v2_iou_dwell_ms": v2_result.iou_dwell_ms_by_object.get(uid, 0.0),
          "v2_co_motion_dwell_ms": v2_result.cm_dwell_ms_by_object.get(uid, 0.0),
          "v2_dwell_ms": v2_result.dwell_ms_by_object.get(uid, 0.0),
          "v2_interacting": int(tracker.is_object_interacting(hand_side, uid)),
          "hand_score": row["hand_score"],
          "obj_score": row["obj_score"],
          "box_source": "detector_preds",
        })

  return pair_rows
