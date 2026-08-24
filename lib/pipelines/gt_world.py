"""v2 scoring with GT world-frame 3D velocities (oracle ego compensation)."""

from data_loaders.loader_hand_poses import LEFT_HAND_INDEX, RIGHT_HAND_INDEX, Handedness

from lib.scoring.co_motion import (
  co_motion_ok,
  co_motion_proximity_ok,
  co_motion_score_xyz,
  slow_manip_ok,
  stationary_ok,
)
from lib.geometry import bbox_to_ltrb
from lib.hot3d_io import get_hand_boxes, get_object_boxes
from lib.motion.object_aabb import ObjectAabbLibrary
from lib.scoring.interaction_v2 import V2DwellTracker
from lib.scoring.pipeline import ScoringConfig, flatten_row_v2, score_pair, v2_proximity_ok, _spatial_only_flat
from lib.motion.world_motion import WorldMotionHistory, se3_translation_xyz
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions


def _handedness_for_side(hand_side: str) -> Handedness:
  return Handedness.Left if hand_side == "left" else Handedness.Right


def _wrist_xyz_world(hand_pose_collection, hand_side: str) -> tuple[float, float, float] | None:
  if hand_pose_collection is None:
    return None
  pose = hand_pose_collection.poses.get(_handedness_for_side(hand_side))
  if pose is None or pose.wrist_pose is None:
    return None
  return se3_translation_xyz(pose.wrist_pose)


def _object_pose_se3(object_pose_collection, object_uid):
  if object_pose_collection is None:
    return None
  pose = object_pose_collection.poses.get(str(object_uid))
  if pose is None or pose.T_world_object is None:
    return None
  return pose.T_world_object


def _object_xyz_world(object_pose_collection, object_uid: int) -> tuple[float, float, float] | None:
  T = _object_pose_se3(object_pose_collection, object_uid)
  if T is None:
    return None
  return se3_translation_xyz(T)


def score_all_pairs_v2_gt_world(
  hand_side: str,
  hand_ltrb,
  hand_xyz,
  object_boxes,
  object_library,
  image_height,
  config: ScoringConfig,
  object_pose_collection,
  hand_world_history: WorldMotionHistory,
  object_world_history: WorldMotionHistory,
  v2_tracker: V2DwellTracker,
  timestamp_ns: int,
  dt_ns: int,
  *,
  co_motion_v_min_m_s: float,
  enable_stationary: bool = False,
  stationary_v_max_m_s: float = 0.015,
  stationary_aabb_d_max_m: float = 0.12,
  stationary_hand_v_max_m_s: float | None = None,
  enable_slow_manip: bool = False,
  slow_manip_v_max_m_s: float = 0.06,
  enable_aabb_co_motion: bool = False,
  co_motion_aabb_d_max_m: float | None = None,
  object_aabb_library: ObjectAabbLibrary | None = None,
) -> tuple[list[dict], dict[int, float], list[dict]]:
  pair_rows = []
  v2_pair_rows = []
  frame_iou = {}
  iou_ok_by_object: dict[int, bool] = {}
  motion_ok_by_object: dict[int, bool] = {}
  visible_uids: set[int] = set()

  hand_motion = hand_world_history.update(hand_side, hand_xyz, dt_ns, timestamp_ns)

  for object_uid, object_entry in object_boxes.items():
    if object_entry.box2d is None or object_entry.visibility_ratio < config.visibility_threshold:
      continue
    object_ltrb = bbox_to_ltrb(object_entry.box2d, image_height, rotate=True)
    if object_ltrb is None:
      continue

    T_world_object = _object_pose_se3(object_pose_collection, object_uid)
    object_xyz = se3_translation_xyz(T_world_object) if T_world_object is not None else None
    if object_xyz is None:
      continue

    visible_uids.add(object_uid)
    metrics = score_pair(hand_ltrb, object_ltrb, config)
    object_motion = object_world_history.update(object_uid, object_xyz, dt_ns, timestamp_ns)
    cm_score = co_motion_score_xyz(
      hand_motion.velocity,
      object_motion.velocity,
      co_motion_v_min_m_s,
    )
    cm_ok = co_motion_ok(cm_score, config.co_motion_threshold)
    iou_ok = v2_proximity_ok(metrics)
    object_name = object_library.object_id_to_name_dict[object_uid]

    aabb_dist = None
    if object_aabb_library is not None:
      aabb_dist = object_aabb_library.surface_distance_m(
        object_uid, hand_xyz, T_world_object
      )

    stat_ok = False
    if enable_stationary:
      stat_ok = stationary_ok(
        hand_motion.velocity,
        object_motion.velocity,
        v_max_m_s=stationary_v_max_m_s,
        aabb_surface_dist_m=aabb_dist,
        aabb_d_max_m=stationary_aabb_d_max_m,
        hand_v_max_m_s=stationary_hand_v_max_m_s,
      )

    slow_ok = False
    if enable_slow_manip:
      slow_ok = slow_manip_ok(
        hand_motion.velocity,
        object_motion.velocity,
        v_manip_max_m_s=slow_manip_v_max_m_s,
        aabb_surface_dist_m=aabb_dist,
        aabb_d_max_m=stationary_aabb_d_max_m,
      )

    cm_aabb_d = (
      co_motion_aabb_d_max_m
      if co_motion_aabb_d_max_m is not None
      else stationary_aabb_d_max_m
    )
    cm_prox_ok = co_motion_proximity_ok(
      iou_ok=iou_ok,
      aabb_surface_dist_m=aabb_dist,
      aabb_d_max_m=cm_aabb_d,
      enable_aabb_co_motion=enable_aabb_co_motion,
    )
    motion_ok = (cm_prox_ok and cm_ok) or stat_ok or slow_ok
    spatial_ok = cm_prox_ok or iou_ok or stat_ok or slow_ok
    frame_ok = motion_ok

    frame_iou[object_uid] = metrics["iou"]
    iou_ok_by_object[object_uid] = spatial_ok
    motion_ok_by_object[object_uid] = motion_ok

    pair_rows.append({
      **metrics,
      "object_uid": object_uid,
      "object_name": object_name,
      "vis_obj": object_entry.visibility_ratio,
    })
    v2_pair_rows.append({
      "object_uid": object_uid,
      "object_name": object_name,
      "co_motion_score": cm_score,
      "co_motion_ok": cm_ok,
      "cm_prox_ok": cm_prox_ok,
      "stationary_ok": stat_ok,
      "slow_manip_ok": slow_ok,
      "motion_ok": motion_ok,
      "frame_ok": frame_ok,
      "iou_ok": iou_ok,
      "aabb_dist_m": aabb_dist if aabb_dist is not None else float("nan"),
      "iou": metrics["iou"],
    })

  object_world_history.decay_offscreen(visible_uids)

  v2_result = v2_tracker.update_hand_frame(
    hand_side,
    iou_ok_by_object,
    motion_ok_by_object,
    dt_ns,
  )
  for row in v2_pair_rows:
    uid = row["object_uid"]
    row["v2_iou_dwell_ms"] = v2_result.iou_dwell_ms_by_object.get(uid, 0.0)
    row["v2_co_motion_dwell_ms"] = v2_result.cm_dwell_ms_by_object.get(uid, 0.0)
    row["v2_dwell_ms"] = v2_result.dwell_ms_by_object.get(uid, 0.0)
    row["v2_interacting"] = v2_tracker.is_object_interacting(hand_side, uid)

  return pair_rows, frame_iou, v2_pair_rows


def score_hand_at_frame_v2_gt_world(
  hand_side: str,
  hand_entry,
  object_boxes,
  object_library,
  image_height,
  config: ScoringConfig,
  hand_pose_collection,
  object_pose_collection,
  hand_world_history: WorldMotionHistory,
  object_world_history: WorldMotionHistory,
  v2_tracker: V2DwellTracker,
  timestamp_ns: int,
  dt_ns: int,
  *,
  co_motion_v_min_m_s: float,
  enable_stationary: bool = False,
  stationary_v_max_m_s: float = 0.015,
  stationary_aabb_d_max_m: float = 0.12,
  stationary_hand_v_max_m_s: float | None = None,
  enable_slow_manip: bool = False,
  slow_manip_v_max_m_s: float = 0.06,
  enable_aabb_co_motion: bool = False,
  co_motion_aabb_d_max_m: float | None = None,
  object_aabb_library: ObjectAabbLibrary | None = None,
) -> tuple[dict | None, list[dict]]:
  hand_ltrb = bbox_to_ltrb(hand_entry.box2d, image_height, rotate=True)
  if hand_ltrb is None:
    v2_tracker.decay_hand(hand_side, dt_ns)
    hand_world_history.decay_key(hand_side)
    return None, []

  hand_xyz = _wrist_xyz_world(hand_pose_collection, hand_side)
  if hand_xyz is None:
    v2_tracker.decay_hand(hand_side, dt_ns)
    hand_world_history.decay_key(hand_side)
    return None, []

  pair_rows, frame_iou, v2_pair_rows = score_all_pairs_v2_gt_world(
    hand_side,
    hand_ltrb,
    hand_xyz,
    object_boxes,
    object_library,
    image_height,
    config,
    object_pose_collection,
    hand_world_history,
    object_world_history,
    v2_tracker,
    timestamp_ns,
    dt_ns,
    co_motion_v_min_m_s=co_motion_v_min_m_s,
    enable_stationary=enable_stationary,
    stationary_v_max_m_s=stationary_v_max_m_s,
    stationary_aabb_d_max_m=stationary_aabb_d_max_m,
    stationary_hand_v_max_m_s=stationary_hand_v_max_m_s,
    enable_slow_manip=enable_slow_manip,
    slow_manip_v_max_m_s=slow_manip_v_max_m_s,
    enable_aabb_co_motion=enable_aabb_co_motion,
    co_motion_aabb_d_max_m=co_motion_aabb_d_max_m,
    object_aabb_library=object_aabb_library,
  )
  if not v2_pair_rows:
    return None, []

  legacy_flat = _spatial_only_flat(pair_rows)
  visible_uids = {r["object_uid"] for r in v2_pair_rows}
  v2_uid = v2_tracker.pick_export_leader(hand_side, frame_iou, visible_uids)
  v2_row = next((r for r in v2_pair_rows if r["object_uid"] == v2_uid), None) if v2_uid else None
  interacting_ids = [
    uid for uid in v2_tracker.interacting_object_ids(hand_side) if uid in visible_uids
  ]
  hand_interacting = len(interacting_ids) > 0

  row = flatten_row_v2(legacy_flat, v2_row, hand_interacting)
  row["v2_motion_frame"] = "gt_world"
  row["v2_stationary_ok"] = bool(v2_row["stationary_ok"]) if v2_row else False
  row["v2_n_interacting"] = len(interacting_ids)
  if v2_row is not None:
    row["v2_co_motion_ok"] = bool(v2_row["motion_ok"])
    row["v2_frame_ok"] = bool(v2_row["frame_ok"])
    row["v2_aabb_dist_m"] = v2_row.get("aabb_dist_m", float("nan"))
  return row, v2_pair_rows


def score_frame_v2_gt_world(
  timestamp_ns,
  frame_index,
  hand_provider,
  object_provider,
  hand_pose_provider,
  object_pose_provider,
  object_library,
  image_height,
  config: ScoringConfig,
  hand_world_history: WorldMotionHistory,
  object_world_history: WorldMotionHistory,
  v2_tracker: V2DwellTracker,
  dt_ns: int,
  *,
  co_motion_v_min_m_s: float,
  enable_stationary: bool = False,
  stationary_v_max_m_s: float = 0.015,
  stationary_aabb_d_max_m: float = 0.12,
  stationary_hand_v_max_m_s: float | None = None,
  enable_slow_manip: bool = False,
  slow_manip_v_max_m_s: float = 0.06,
  enable_aabb_co_motion: bool = False,
  co_motion_aabb_d_max_m: float | None = None,
  object_aabb_library: ObjectAabbLibrary | None = None,
) -> tuple[list[dict], list[dict]]:
  hand_result = get_hand_boxes(timestamp_ns, hand_provider)
  object_result = get_object_boxes(timestamp_ns, object_provider)
  if hand_result is None or object_result is None:
    return [], []

  hand_pose_with_dt = hand_pose_provider.get_pose_at_timestamp(
    timestamp_ns,
    TimeQueryOptions.CLOSEST,
    TimeDomain.TIME_CODE,
  )
  hand_pose_collection = (
    hand_pose_with_dt.pose3d_collection if hand_pose_with_dt is not None else None
  )

  object_pose_with_dt = object_pose_provider.get_pose_at_timestamp(
    timestamp_ns,
    TimeQueryOptions.CLOSEST,
    TimeDomain.TIME_CODE,
  )
  object_pose_collection = (
    object_pose_with_dt.pose3d_collection if object_pose_with_dt is not None else None
  )

  rows = []
  pair_export_rows = []
  for hand_uid in (LEFT_HAND_INDEX, RIGHT_HAND_INDEX):
    hand_side = "left" if hand_uid == LEFT_HAND_INDEX else "right"
    hand_entry = hand_result["boxes"].get(hand_uid)
    if hand_entry is None:
      v2_tracker.decay_hand(hand_side, dt_ns)
      hand_world_history.decay_key(hand_side)
      continue
    if hand_entry.box2d is None or hand_entry.visibility_ratio < config.visibility_threshold:
      v2_tracker.decay_hand(hand_side, dt_ns)
      hand_world_history.decay_key(hand_side)
      continue

    best, v2_pair_rows = score_hand_at_frame_v2_gt_world(
      hand_side,
      hand_entry,
      object_result["boxes"],
      object_library,
      image_height,
      config,
      hand_pose_collection,
      object_pose_collection,
      hand_world_history,
      object_world_history,
      v2_tracker,
      timestamp_ns,
      dt_ns,
      co_motion_v_min_m_s=co_motion_v_min_m_s,
      enable_stationary=enable_stationary,
      stationary_v_max_m_s=stationary_v_max_m_s,
      stationary_aabb_d_max_m=stationary_aabb_d_max_m,
      stationary_hand_v_max_m_s=stationary_hand_v_max_m_s,
      enable_slow_manip=enable_slow_manip,
      slow_manip_v_max_m_s=slow_manip_v_max_m_s,
      enable_aabb_co_motion=enable_aabb_co_motion,
      co_motion_aabb_d_max_m=co_motion_aabb_d_max_m,
      object_aabb_library=object_aabb_library,
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
    for pair in v2_pair_rows:
      pair_export_rows.append({
        "timestamp_ns": timestamp_ns,
        "frame_index": frame_index,
        "hand_side": hand_side,
        "object_uid": pair["object_uid"],
        "object_name": pair["object_name"],
        "iou": pair["iou"],
        "v2_aabb_dist_m": pair.get("aabb_dist_m", float("nan")),
        "v2_iou_ok": pair["iou_ok"],
        "v2_cm_prox_ok": pair.get("cm_prox_ok", pair["iou_ok"]),
        "v2_co_motion_score": pair["co_motion_score"],
        "v2_co_motion_ok": pair["co_motion_ok"],
        "v2_stationary_ok": pair["stationary_ok"],
        "v2_slow_manip_ok": pair.get("slow_manip_ok", False),
        "v2_motion_ok": pair["motion_ok"],
        "v2_frame_ok": pair["frame_ok"],
        "v2_iou_dwell_ms": pair["v2_iou_dwell_ms"],
        "v2_co_motion_dwell_ms": pair["v2_co_motion_dwell_ms"],
        "v2_dwell_ms": pair["v2_dwell_ms"],
        "v2_interacting": pair["v2_interacting"],
      })
  return rows, pair_export_rows
