"""Temporal dwell tracking for legacy v1 (TotTracker)."""

import numpy as np
from dataclasses import dataclass

from lib.scoring.motion import HandDynamicHistory, HandMotionState

__all__ = ["HandDynamicHistory", "HandMotionState", "TotFrameResult", "TotTracker"]


@dataclass
class TotFrameResult:
  tot_ms_by_object: dict[int, float]
  leader_id: int | None


class TotTracker:
  def __init__(
    self,
    engagement_threshold: float = 0.35,
    tot_min_ms: float = 150.0,
    gap_tolerance_ms: float = 130.0,
    velocity_threshold: float = 150.0,
    norm_variance_threshold: float = 0.002,
    manipulation_scale: float = 1.2,
    idle_scale: float = 0.8,
  ):
    self._tot_ms: dict[tuple[str, int], float] = {}
    self._missed_ms: dict[tuple[str, int], float] = {}
    self.engagement_threshold = engagement_threshold
    self.tot_min_ms = tot_min_ms
    self.gap_tolerance_ms = gap_tolerance_ms
    self.velocity_threshold = velocity_threshold
    self.norm_variance_threshold = norm_variance_threshold
    self.manipulation_scale = manipulation_scale
    self.idle_scale = idle_scale

  def _hand_speed(self, hand_velocity: tuple[float, float] | None) -> float:
    if hand_velocity is None:
      return 0.0
    return float(np.hypot(hand_velocity[0], hand_velocity[1]))

  def _adjust_engagement(
    self,
    raw_engagement: float,
    hand_velocity: tuple[float, float] | None,
    hand_norm_dim_variance: float,
  ) -> float:
    """Scale raw engagement from hand kinematics before dwell accumulation."""
    is_transit = self._hand_speed(hand_velocity) > self.velocity_threshold
    is_manipulating = hand_norm_dim_variance > self.norm_variance_threshold

    if is_transit:
      return raw_engagement
    if is_manipulating:
      return raw_engagement * self.manipulation_scale
    return raw_engagement * self.idle_scale

  def _accumulate(self, hand_side: str, object_id: int, dt_ms: float) -> None:
    key = (hand_side, object_id)
    self._tot_ms[key] = self._tot_ms.get(key, 0.0) + dt_ms
    self._missed_ms[key] = 0.0

  def _gap_decay(self, hand_side: str, object_id: int, dt_ms: float) -> None:
    key = (hand_side, object_id)
    self._missed_ms[key] = self._missed_ms.get(key, 0.0) + dt_ms
    if self._missed_ms[key] >= self.gap_tolerance_ms:
      self._tot_ms[key] = 0.0
      self._missed_ms[key] = 0.0

  def _tracked_object_ids(self, hand_side: str) -> set[int]:
    ids: set[int] = set()
    for side, object_id in self._tot_ms:
      if side == hand_side:
        ids.add(object_id)
    for side, object_id in self._missed_ms:
      if side == hand_side:
        ids.add(object_id)
    return ids

  def _gap_decay_offscreen(
    self,
    hand_side: str,
    visible_object_ids: set[int],
    dt_ms: float,
  ) -> None:
    """Gap-decay tracked objects for this hand that are not visible this frame."""
    for object_id in self._tracked_object_ids(hand_side):
      if object_id not in visible_object_ids:
        self._gap_decay(hand_side, object_id, dt_ms)

  def _pick_frame_winner(
    self, adjusted_by_object: dict[int, float]
  ) -> int | None:
    if not adjusted_by_object:
      return None
    return max(adjusted_by_object, key=lambda oid: (adjusted_by_object[oid], oid))

  def update_hand_frame(
    self,
    hand_side: str,
    frame_engagements: dict[int, float],
    hand_velocity: tuple[float, float] | None,
    hand_norm_dim_variance: float,
    dt_ns: int,
  ) -> TotFrameResult:
    """Update dwell for one hand frame using middle-ground winner accumulation."""
    dt_ms = dt_ns / 1e6
    adjusted_by_object = {
      object_id: self._adjust_engagement(
        raw_engagement, hand_velocity, hand_norm_dim_variance
      )
      for object_id, raw_engagement in frame_engagements.items()
    }
    visible_object_ids = set(adjusted_by_object.keys())

    if not adjusted_by_object:
      self._gap_decay_offscreen(hand_side, visible_object_ids, dt_ms)
      return TotFrameResult(tot_ms_by_object={}, leader_id=None)

    frame_winner_id = self._pick_frame_winner(adjusted_by_object)
    frame_winner_engagement = adjusted_by_object[frame_winner_id]

    if frame_winner_engagement >= self.engagement_threshold:
      for object_id in adjusted_by_object:
        if object_id == frame_winner_id:
          self._accumulate(hand_side, object_id, dt_ms)
        else:
          self._gap_decay(hand_side, object_id, dt_ms)
    else:
      for object_id in adjusted_by_object:
        self._gap_decay(hand_side, object_id, dt_ms)

    self._gap_decay_offscreen(hand_side, visible_object_ids, dt_ms)

    tot_ms_by_object = {
      object_id: self.get_tot_ms(hand_side, object_id)
      for object_id in adjusted_by_object
    }
    leader_id = self._pick_frame_winner(tot_ms_by_object)

    return TotFrameResult(
      tot_ms_by_object=tot_ms_by_object,
      leader_id=leader_id,
    )

  def update(
    self,
    hand_side: str,
    object_id: int,
    engagement: float,
    dt_ns: int,
    hand_velocity: tuple[float, float] | None = None,
    hand_norm_dim_variance: float = 0.0,
  ) -> float:
    """Update one hand-object pair. Prefer ``update_hand_frame`` per hand per frame."""
    result = self.update_hand_frame(
      hand_side,
      {object_id: engagement},
      hand_velocity,
      hand_norm_dim_variance,
      dt_ns,
    )
    return result.tot_ms_by_object.get(object_id, 0.0)

  def get_tot_ms(self, hand_side: str, object_id: int) -> float:
    return self._tot_ms.get((hand_side, object_id), 0.0)

  def decay_hand(self, hand_side: str, dt_ns: int) -> None:
    """Apply gap decay to all tracked objects for a hand when it is not visible this frame."""
    dt_ms = dt_ns / 1e6
    for object_id in self._tracked_object_ids(hand_side):
      self._gap_decay(hand_side, object_id, dt_ms)

  def best_object(
    self,
    hand_side: str,
    frame_iou: dict[int, float] | None = None,
    candidates_ids: set[int] | None = None,
  ) -> int | None:
    """Return the object with max cumulative dwell; tie-break on frame IoU."""
    best_tot = 0.0
    best_id = None
    for (side, object_id), tot_ms in self._tot_ms.items():
      if side != hand_side:
        continue
      if candidates_ids is not None and object_id not in candidates_ids:
        continue
      if tot_ms > best_tot:
        best_tot = tot_ms
        best_id = object_id
      elif tot_ms == best_tot and tot_ms > 0:
        if frame_iou is not None and frame_iou.get(object_id, 0) > frame_iou.get(best_id, 0):
          best_id = object_id
    return best_id

  def is_interacting(self, hand_side: str, object_id: int | None) -> bool:
    """True only for the export leader with enough cumulative dwell."""
    if object_id is None:
      return False
    if self.get_tot_ms(hand_side, object_id) < self.tot_min_ms:
      return False
    return object_id == self.best_object(hand_side)
