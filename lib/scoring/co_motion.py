"""Hand–object co-motion: velocity alignment with minimum-speed gating."""

from __future__ import annotations

from typing import Optional

import numpy as np


def speed(velocity: tuple[float, float] | None) -> float:
  """XY speed (image / legacy 2D motion frames)."""
  if velocity is None:
    return 0.0
  return float(np.hypot(velocity[0], velocity[1]))


def world_speed(velocity: tuple[float, ...] | None) -> float:
  """L2 speed for 2D or 3D world-frame velocity tuples."""
  if velocity is None:
    return 0.0
  return float(np.linalg.norm(velocity))


def co_motion_score(
  hand_velocity: tuple[float, float] | None,
  object_velocity: tuple[float, float] | None,
  v_min: float,
) -> float:
  """
  max(0, cos(hand_vel, object_vel)) in XY; 0 when either XY speed is below v_min.

  Used by image_2d / non-GT paths. Prefer co_motion_score_xyz for gt_world.
  """
  if hand_velocity is None or object_velocity is None or v_min <= 0:
    return 0.0
  hand_speed = speed(hand_velocity)
  object_speed = speed(object_velocity)
  if hand_speed < v_min or object_speed < v_min:
    return 0.0
  dot = hand_velocity[0] * object_velocity[0] + hand_velocity[1] * object_velocity[1]
  denom = hand_speed * object_speed
  if denom <= 1e-9:
    return 0.0
  return max(0.0, dot / denom)


def co_motion_score_xyz(
  hand_velocity: tuple[float, ...] | None,
  object_velocity: tuple[float, ...] | None,
  v_min: float,
) -> float:
  """
  Full 3D co-motion for gt_world: max(0, cos) with L2 speed floor on both.

  Lifts / pours that are mostly Z are visible here (unlike XY-only).
  """
  if hand_velocity is None or object_velocity is None or v_min <= 0:
    return 0.0
  hand_v = np.asarray(hand_velocity, dtype=float).reshape(-1)
  object_v = np.asarray(object_velocity, dtype=float).reshape(-1)
  if hand_v.size < 3 or object_v.size < 3:
    return 0.0
  hand_speed = float(np.linalg.norm(hand_v[:3]))
  object_speed = float(np.linalg.norm(object_v[:3]))
  if hand_speed < v_min or object_speed < v_min:
    return 0.0
  denom = hand_speed * object_speed
  if denom <= 1e-9:
    return 0.0
  dot = float(np.dot(hand_v[:3], object_v[:3]))
  return max(0.0, dot / denom)


def co_motion_ok(score: float, threshold: float) -> bool:
  return score >= threshold


def co_motion_proximity_ok(
  *,
  iou_ok: bool,
  aabb_surface_dist_m: float | None,
  aabb_d_max_m: float,
  enable_aabb_co_motion: bool,
) -> bool:
  """
  Proximity gate for the moving co-motion branch.

  Default (enable_aabb_co_motion=False): 2D IoU.
  Experimental: wrist within AABB surface distance (same d_max as stationary).
  """
  if enable_aabb_co_motion:
    if aabb_surface_dist_m is None or aabb_d_max_m <= 0:
      return False
    return aabb_surface_dist_m <= aabb_d_max_m
  return bool(iou_ok)


def stationary_ok(
  hand_velocity: tuple[float, ...] | None,
  object_velocity: tuple[float, ...] | None,
  *,
  v_max_m_s: float,
  aabb_surface_dist_m: float | None,
  aabb_d_max_m: float,
  hand_v_max_m_s: float | None = None,
) -> bool:
  """
  Class-free stationary gate: object slow + wrist within AABB surface distance.

  ``hand_v_max_m_s`` defaults to ``v_max_m_s`` (symmetric). Experimental presets
  may loosen hand only while object stays the stillness anchor.
  """
  if hand_velocity is None or object_velocity is None or v_max_m_s <= 0:
    return False
  if aabb_surface_dist_m is None or aabb_d_max_m <= 0:
    return False
  if aabb_surface_dist_m > aabb_d_max_m:
    return False
  hand_cap = v_max_m_s if hand_v_max_m_s is None else hand_v_max_m_s
  return (
    world_speed(hand_velocity) < hand_cap
    and world_speed(object_velocity) < v_max_m_s
  )


def slow_manip_ok(
  hand_velocity: tuple[float, ...] | None,
  object_velocity: tuple[float, ...] | None,
  *,
  v_manip_max_m_s: float,
  aabb_surface_dist_m: float | None,
  aabb_d_max_m: float,
) -> bool:
  """
  Opt-in near-hold / slow-rotate mid-band: AABB contact + both speeds below
  a higher manip ceiling (no co-motion cosine). Default path leaves this off.
  """
  if hand_velocity is None or object_velocity is None or v_manip_max_m_s <= 0:
    return False
  if aabb_surface_dist_m is None or aabb_d_max_m <= 0:
    return False
  if aabb_surface_dist_m > aabb_d_max_m:
    return False
  return (
    world_speed(hand_velocity) < v_manip_max_m_s
    and world_speed(object_velocity) < v_manip_max_m_s
  )
