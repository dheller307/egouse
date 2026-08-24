"""Scoring pipeline: legacy (geometry → engagement → ToT) and v2 (spatial + co-motion)."""

from dataclasses import dataclass

from lib.scoring.co_motion import co_motion_ok, co_motion_score
from lib.geometry import (
  approach_alignment,
  bbox_to_ltrb,
  bbox_iou,
  distance_confidence,
  engagement_confidence,
  norm_center_distance,
  spatial_confidence,
)
from lib.scoring.interaction_v2 import V2DwellTracker
from lib.scoring.motion import HandDynamicHistory, ObjectDynamicHistory
from lib.scoring.temporal import HandMotionState, TotTracker


@dataclass
class ScoringConfig:
  visibility_threshold: float
  iou_threshold: float
  dist_threshold: float
  spatial_threshold: float
  k: float
  w_spatial: float
  w_approach: float
  engagement_threshold: float
  tot_min_ms: float
  max_history_len: int
  smooth_frames: int
  min_var_frames: int
  gap_reset_ms: float
  gap_tolerance_ms: float
  velocity_threshold: float
  norm_variance_threshold: float
  manipulation_scale: float
  idle_scale: float
  # v2 knobs (ignored in legacy mode)
  co_motion_threshold: float = 0.5
  co_motion_v_min: float = 30.0
  v2_iou_dwell_ms: float = 99.0
  v2_co_motion_dwell_ms: float = 66.0
  v2_iou_gap_tolerance_ms: float = 130.0
  v2_co_motion_gap_tolerance_ms: float = 220.0


def pick_best_pair(pair_rows, key_fn) -> dict | None:
  """Pick the row with the highest value for the given key function."""
  if not pair_rows:
    return None
  return max(pair_rows, key=lambda r: (key_fn(r), r["object_uid"]))


def flatten_row(
  iou_row: dict | None,
  dist_row: dict | None,
  spatial_row: dict | None,
  engagement_row: dict | None,
  tot_row: dict | None,
  tot_interacting: bool,
) -> dict:
  return {
    "iou_object_name": iou_row["object_name"],
    "iou_score": iou_row["iou"],
    "iou_interacting": iou_row["iou_interacting"],
    "dist_object_name": dist_row["object_name"],
    "dist_score": dist_row["dist_conf"],
    "dist_interacting": dist_row["dist_interacting"],
    "spatial_object_name": spatial_row["object_name"],
    "spatial_score": spatial_row["spatial_conf"],
    "spatial_interacting": spatial_row["spatial_interacting"],
    "engagement_object_name": engagement_row["object_name"],
    "engagement_score": engagement_row["engagement_score"],
    "engagement_interacting": engagement_row["engagement_interacting"],
    "tot_object_name": tot_row["object_name"] if tot_row else "none",
    "tot_interacting": tot_interacting,
    "tot_ms": tot_row["tot_ms"] if tot_row else 0.0,
  }


def flatten_row_v2(
  legacy_row: dict,
  v2_row: dict | None,
  v2_interacting: bool,
) -> dict:
  row = dict(legacy_row)
  if v2_row is None:
    row.update({
      "v2_object_name": "none",
      "v2_co_motion_score": 0.0,
      "v2_co_motion_ok": False,
      "v2_iou_ok": False,
      "v2_frame_ok": False,
      "v2_iou_dwell_ms": 0.0,
      "v2_co_motion_dwell_ms": 0.0,
      "v2_dwell_ms": 0.0,
      "v2_interacting": v2_interacting,
    })
  else:
    row.update({
      "v2_object_name": v2_row["object_name"],
      "v2_co_motion_score": v2_row["co_motion_score"],
      "v2_co_motion_ok": v2_row["co_motion_ok"],
      "v2_iou_ok": v2_row["iou_ok"],
      "v2_frame_ok": v2_row["frame_ok"],
      "v2_iou_dwell_ms": v2_row["v2_iou_dwell_ms"],
      "v2_co_motion_dwell_ms": v2_row["v2_co_motion_dwell_ms"],
      "v2_dwell_ms": v2_row["v2_dwell_ms"],
      "v2_interacting": v2_interacting,
    })
  return row


def score_pair(hand_ltrb, object_ltrb, config: ScoringConfig):
  iou = bbox_iou(hand_ltrb, object_ltrb)
  d_norm = norm_center_distance(hand_ltrb, object_ltrb)
  dist_score = distance_confidence(hand_ltrb, object_ltrb, config.k)
  spatial_score = spatial_confidence(iou, dist_score)
  return {
    "iou": iou,
    "d_norm": d_norm,
    "dist_conf": dist_score,
    "spatial_conf": spatial_score,
    "iou_interacting": iou > config.iou_threshold,
    "dist_interacting": dist_score > config.dist_threshold,
    "spatial_interacting": spatial_score > config.spatial_threshold,
  }


def v2_proximity_ok(metrics: dict) -> bool:
  """v2 spatial gate: IoU only (distance excluded from interaction decision)."""
  return metrics["iou_interacting"]


def score_all_pairs(
  hand_side: str,
  hand_ltrb,
  motion: HandMotionState,
  object_boxes,
  object_library,
  image_height,
  config: ScoringConfig,
  tracker: TotTracker,
  dt_ns: int,
) -> tuple[list[dict], dict[int, float]]:
  pair_rows = []
  frame_iou = {}
  frame_engagements: dict[int, float] = {}

  for object_uid, object_entry in object_boxes.items():
    if object_entry.box2d is None or object_entry.visibility_ratio < config.visibility_threshold:
      continue
    object_ltrb = bbox_to_ltrb(object_entry.box2d, image_height, rotate=True)
    if object_ltrb is None:
      continue
    metrics = score_pair(hand_ltrb, object_ltrb, config)
    approach_score = approach_alignment(motion.velocity, hand_ltrb, object_ltrb)
    engagement_score = engagement_confidence(
      metrics["spatial_conf"], approach_score, config.w_spatial, config.w_approach
    )
    frame_engagements[object_uid] = engagement_score
    frame_iou[object_uid] = metrics["iou"]
    pair_rows.append({
      **metrics,
      "object_uid": object_uid,
      "object_name": object_library.object_id_to_name_dict[object_uid],
      "vis_obj": object_entry.visibility_ratio,
      "approach_score": approach_score,
      "engagement_score": engagement_score,
      "engagement_interacting": engagement_score > config.engagement_threshold,
    })

  tot_result = tracker.update_hand_frame(
    hand_side,
    frame_engagements,
    motion.velocity,
    motion.norm_dim_variance,
    dt_ns,
  )
  for row in pair_rows:
    row["tot_ms"] = tot_result.tot_ms_by_object.get(row["object_uid"], 0.0)

  return pair_rows, frame_iou


def score_all_pairs_v2(
  hand_side: str,
  hand_ltrb,
  hand_motion: HandMotionState,
  object_boxes,
  object_library,
  image_height,
  config: ScoringConfig,
  object_history: ObjectDynamicHistory,
  v2_tracker: V2DwellTracker,
  timestamp_ns: int,
  dt_ns: int,
) -> tuple[list[dict], dict[int, float], list[dict]]:
  pair_rows = []
  v2_pair_rows = []
  frame_iou = {}
  iou_ok_by_object: dict[int, bool] = {}
  cm_ok_by_object: dict[int, bool] = {}
  co_motion_by_object: dict[int, float] = {}
  visible_uids: set[int] = set()

  for object_uid, object_entry in object_boxes.items():
    if object_entry.box2d is None or object_entry.visibility_ratio < config.visibility_threshold:
      continue
    object_ltrb = bbox_to_ltrb(object_entry.box2d, image_height, rotate=True)
    if object_ltrb is None:
      continue
    visible_uids.add(object_uid)
    metrics = score_pair(hand_ltrb, object_ltrb, config)
    object_motion = object_history.update(object_uid, object_ltrb, dt_ns, timestamp_ns)
    cm_score = co_motion_score(
      hand_motion.velocity,
      object_motion.velocity,
      config.co_motion_v_min,
    )
    cm_ok = co_motion_ok(cm_score, config.co_motion_threshold)
    iou_ok = v2_proximity_ok(metrics)
    frame_ok = iou_ok and cm_ok

    frame_iou[object_uid] = metrics["iou"]
    iou_ok_by_object[object_uid] = iou_ok
    cm_ok_by_object[object_uid] = cm_ok
    co_motion_by_object[object_uid] = cm_score

    pair_rows.append({
      **metrics,
      "object_uid": object_uid,
      "object_name": object_library.object_id_to_name_dict[object_uid],
      "vis_obj": object_entry.visibility_ratio,
    })
    v2_pair_rows.append({
      "object_uid": object_uid,
      "object_name": object_library.object_id_to_name_dict[object_uid],
      "co_motion_score": cm_score,
      "co_motion_ok": cm_ok,
      "frame_ok": frame_ok,
      "iou_ok": iou_ok,
      "iou": metrics["iou"],
    })

  object_history.decay_offscreen(visible_uids)

  v2_result = v2_tracker.update_hand_frame(
    hand_side,
    iou_ok_by_object,
    cm_ok_by_object,
    dt_ns,
  )
  for row in v2_pair_rows:
    uid = row["object_uid"]
    row["v2_iou_dwell_ms"] = v2_result.iou_dwell_ms_by_object.get(uid, 0.0)
    row["v2_co_motion_dwell_ms"] = v2_result.cm_dwell_ms_by_object.get(uid, 0.0)
    row["v2_dwell_ms"] = v2_result.dwell_ms_by_object.get(uid, 0.0)

  return pair_rows, frame_iou, v2_pair_rows


def score_hand_at_frame(
  hand_side: str,
  hand_entry,
  object_boxes,
  object_library,
  image_height,
  config: ScoringConfig,
  history: HandDynamicHistory,
  tracker: TotTracker,
  timestamp_ns: int,
  dt_ns: int,
) -> dict | None:
  hand_ltrb = bbox_to_ltrb(hand_entry.box2d, image_height, rotate=True)
  if hand_ltrb is None:
    tracker.decay_hand(hand_side, dt_ns)
    return None
  motion = history.update(hand_side, hand_ltrb, dt_ns, timestamp_ns)
  pair_rows, frame_iou = score_all_pairs(
    hand_side,
    hand_ltrb,
    motion,
    object_boxes,
    object_library,
    image_height,
    config,
    tracker,
    dt_ns,
  )
  if len(pair_rows) == 0:
    return None
  visible_uids = {r["object_uid"] for r in pair_rows}
  iou_row = pick_best_pair(pair_rows, lambda r: r["iou"])
  dist_row = pick_best_pair(pair_rows, lambda r: r["dist_conf"])
  spatial_row = pick_best_pair(pair_rows, lambda r: r["spatial_conf"])
  engagement_row = pick_best_pair(
    pair_rows, lambda r: r["engagement_score"]
  )
  tot_uid = tracker.best_object(hand_side, frame_iou, visible_uids)
  tot_row = next((r for r in pair_rows if r["object_uid"] == tot_uid), None) if tot_uid else None
  return flatten_row(
    iou_row,
    dist_row,
    spatial_row,
    engagement_row,
    tot_row,
    tracker.is_interacting(hand_side, tot_uid),
  )


def _spatial_only_flat(pair_rows: list[dict]) -> dict:
  """Minimal legacy columns from spatial pair rows (v2 mode without full legacy stack)."""
  iou_row = pick_best_pair(pair_rows, lambda r: r["iou"])
  dist_row = pick_best_pair(pair_rows, lambda r: r["dist_conf"])
  spatial_row = pick_best_pair(pair_rows, lambda r: r["spatial_conf"])
  return {
    "iou_object_name": iou_row["object_name"],
    "iou_score": iou_row["iou"],
    "iou_interacting": iou_row["iou_interacting"],
    "dist_object_name": dist_row["object_name"],
    "dist_score": dist_row["dist_conf"],
    "dist_interacting": dist_row["dist_interacting"],
    "spatial_object_name": spatial_row["object_name"],
    "spatial_score": spatial_row["spatial_conf"],
    "spatial_interacting": spatial_row["spatial_interacting"],
    "engagement_object_name": "none",
    "engagement_score": 0.0,
    "engagement_interacting": False,
    "tot_object_name": "none",
    "tot_interacting": False,
    "tot_ms": 0.0,
  }


def score_hand_at_frame_v2(
  hand_side: str,
  hand_entry,
  object_boxes,
  object_library,
  image_height,
  config: ScoringConfig,
  hand_history: HandDynamicHistory,
  object_history: ObjectDynamicHistory,
  legacy_tracker: TotTracker,
  v2_tracker: V2DwellTracker,
  timestamp_ns: int,
  dt_ns: int,
  include_legacy: bool = True,
) -> dict | None:
  hand_ltrb = bbox_to_ltrb(hand_entry.box2d, image_height, rotate=True)
  if hand_ltrb is None:
    legacy_tracker.decay_hand(hand_side, dt_ns)
    v2_tracker.decay_hand(hand_side, dt_ns)
    return None

  hand_motion = hand_history.update(hand_side, hand_ltrb, dt_ns, timestamp_ns)

  legacy_flat = None
  if include_legacy:
    pair_rows, frame_iou = score_all_pairs(
      hand_side,
      hand_ltrb,
      hand_motion,
      object_boxes,
      object_library,
      image_height,
      config,
      legacy_tracker,
      dt_ns,
    )
    if len(pair_rows) == 0:
      return None
    visible_uids = {r["object_uid"] for r in pair_rows}
    iou_row = pick_best_pair(pair_rows, lambda r: r["iou"])
    dist_row = pick_best_pair(pair_rows, lambda r: r["dist_conf"])
    spatial_row = pick_best_pair(pair_rows, lambda r: r["spatial_conf"])
    engagement_row = pick_best_pair(pair_rows, lambda r: r["engagement_score"])
    tot_uid = legacy_tracker.best_object(hand_side, frame_iou, visible_uids)
    tot_row = (
      next((r for r in pair_rows if r["object_uid"] == tot_uid), None)
      if tot_uid
      else None
    )
    legacy_flat = flatten_row(
      iou_row,
      dist_row,
      spatial_row,
      engagement_row,
      tot_row,
      legacy_tracker.is_interacting(hand_side, tot_uid),
    )

  pair_rows, frame_iou, v2_pair_rows = score_all_pairs_v2(
    hand_side,
    hand_ltrb,
    hand_motion,
    object_boxes,
    object_library,
    image_height,
    config,
    object_history,
    v2_tracker,
    timestamp_ns,
    dt_ns,
  )
  if not v2_pair_rows:
    return None

  if legacy_flat is None:
    legacy_flat = _spatial_only_flat(pair_rows)

  visible_uids = {r["object_uid"] for r in v2_pair_rows}
  v2_uid = v2_tracker.best_object(hand_side, frame_iou, visible_uids)
  v2_row = next((r for r in v2_pair_rows if r["object_uid"] == v2_uid), None) if v2_uid else None

  return flatten_row_v2(
    legacy_flat,
    v2_row,
    v2_tracker.is_interacting(hand_side, v2_uid),
  )


def score_frame(
  timestamp_ns,
  frame_index,
  hand_provider,
  object_provider,
  object_library,
  image_height,
  config: ScoringConfig,
  history: HandDynamicHistory,
  tracker: TotTracker,
  dt_ns: int,
) -> list[dict]:
  from data_loaders.loader_hand_poses import LEFT_HAND_INDEX, RIGHT_HAND_INDEX

  from lib.hot3d_io import get_hand_boxes, get_object_boxes

  hand_result = get_hand_boxes(timestamp_ns, hand_provider)
  object_result = get_object_boxes(timestamp_ns, object_provider)
  if hand_result is None or object_result is None:
    return []

  rows = []
  for hand_uid in (LEFT_HAND_INDEX, RIGHT_HAND_INDEX):
    hand_side = "left" if hand_uid == LEFT_HAND_INDEX else "right"
    hand_entry = hand_result["boxes"][hand_uid]
    if hand_entry.box2d is None or hand_entry.visibility_ratio < config.visibility_threshold:
      tracker.decay_hand(hand_side, dt_ns)
      continue
    best = score_hand_at_frame(
      hand_side,
      hand_entry,
      object_result["boxes"],
      object_library,
      image_height,
      config,
      history,
      tracker,
      timestamp_ns,
      dt_ns,
    )
    if best is None:
      continue
    rows.append({
      "timestamp_ns": timestamp_ns,
      "frame_index": frame_index,
      "hand_side": hand_side,
      "vis_hand": hand_entry.visibility_ratio,
      **best,
    })
  return rows


def score_frame_v2(
  timestamp_ns,
  frame_index,
  hand_provider,
  object_provider,
  object_library,
  image_height,
  config: ScoringConfig,
  hand_history: HandDynamicHistory,
  object_history: ObjectDynamicHistory,
  legacy_tracker: TotTracker,
  v2_tracker: V2DwellTracker,
  dt_ns: int,
  include_legacy: bool = True,
) -> list[dict]:
  from data_loaders.loader_hand_poses import LEFT_HAND_INDEX, RIGHT_HAND_INDEX

  from lib.hot3d_io import get_hand_boxes, get_object_boxes

  hand_result = get_hand_boxes(timestamp_ns, hand_provider)
  object_result = get_object_boxes(timestamp_ns, object_provider)
  if hand_result is None or object_result is None:
    return []

  rows = []
  for hand_uid in (LEFT_HAND_INDEX, RIGHT_HAND_INDEX):
    hand_side = "left" if hand_uid == LEFT_HAND_INDEX else "right"
    hand_entry = hand_result["boxes"][hand_uid]
    if hand_entry.box2d is None or hand_entry.visibility_ratio < config.visibility_threshold:
      legacy_tracker.decay_hand(hand_side, dt_ns)
      v2_tracker.decay_hand(hand_side, dt_ns)
      continue
    best = score_hand_at_frame_v2(
      hand_side,
      hand_entry,
      object_result["boxes"],
      object_library,
      image_height,
      config,
      hand_history,
      object_history,
      legacy_tracker,
      v2_tracker,
      timestamp_ns,
      dt_ns,
      include_legacy=include_legacy,
    )
    if best is None:
      continue
    rows.append({
      "timestamp_ns": timestamp_ns,
      "frame_index": frame_index,
      "hand_side": hand_side,
      "vis_hand": hand_entry.visibility_ratio,
      **best,
    })
  return rows
