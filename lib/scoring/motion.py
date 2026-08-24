"""Hand and object bbox velocity from rolling center history."""

import numpy as np
from collections import deque
from dataclasses import dataclass

from lib.geometry import bbox_center, hand_velocity_px_per_s

_EPS = 1e-6


@dataclass
class MotionState:
  velocity: tuple[float, float] | None
  dim_variance: float
  norm_dim_variance: float
  is_ready: bool


# Backward-compatible alias used by legacy pipeline / TotTracker.
HandMotionState = MotionState


class _BBoxDynamicHistory:
  """Rolling bbox buffer with smoothed center velocity (shared hand/object logic)."""

  def __init__(
    self,
    max_history_len: int = 15,
    smooth_frames: int = 5,
    min_var_frames: int = 5,
    gap_reset_ms: float = 200.0,
  ):
    self._max_history_len = max_history_len
    self._smooth_frames = smooth_frames
    self._min_var_frames = min_var_frames
    self._gap_reset_ms = gap_reset_ms
    self._history: dict[int | str, deque] = {}
    self._prev_smooth: dict[int | str, tuple[float, float]] = {}
    self._last_seen_ns: dict[int | str, int] = {}

  def _empty_state(self) -> MotionState:
    return MotionState(
      velocity=None,
      dim_variance=0.0,
      norm_dim_variance=0.0,
      is_ready=False,
    )

  def _clear_motion_buffer(self, key: int | str) -> None:
    self._prev_smooth.pop(key, None)
    if key in self._history:
      self._history[key].clear()

  def _reset_key(self, key: int | str) -> None:
    self._clear_motion_buffer(key)
    self._last_seen_ns.pop(key, None)

  def update(
    self,
    key: int | str,
    ltrb,
    dt_ns: int,
    timestamp_ns: int,
  ) -> MotionState:
    if ltrb is None:
      self._reset_key(key)
      return self._empty_state()

    last_seen_ns = self._last_seen_ns.get(key)
    if last_seen_ns is not None:
      gap_ms = (timestamp_ns - last_seen_ns) / 1e6
      if gap_ms > self._gap_reset_ms:
        self._clear_motion_buffer(key)

    if key not in self._history:
      self._history[key] = deque(maxlen=self._max_history_len)

    self._history[key].append(ltrb)
    history_window = self._history[key]

    smooth_window = list(history_window)[-self._smooth_frames:]
    centers = []
    for box in smooth_window:
      center = bbox_center(box)
      if center is not None:
        centers.append(center)
    if not centers:
      return self._empty_state()

    avg_x = sum(c[0] for c in centers) / len(centers)
    avg_y = sum(c[1] for c in centers) / len(centers)
    curr_smooth_center = (avg_x, avg_y)

    is_ready = len(history_window) >= self._smooth_frames
    prev_smooth = self._prev_smooth.get(key)
    velocity = None
    if is_ready and prev_smooth is not None:
      velocity = hand_velocity_px_per_s(prev_smooth, curr_smooth_center, dt_ns)
    self._prev_smooth[key] = curr_smooth_center

    dim_variance = 0.0
    norm_dim_variance = 0.0
    if len(history_window) >= self._min_var_frames:
      widths = [box[2] - box[0] for box in history_window]
      heights = [box[3] - box[1] for box in history_window]
      dim_variance = float(np.var(widths) + np.var(heights))
      w = ltrb[2] - ltrb[0]
      h = ltrb[3] - ltrb[1]
      scale = max(w * h, _EPS)
      norm_dim_variance = dim_variance / scale

    self._last_seen_ns[key] = timestamp_ns

    return MotionState(
      velocity=velocity,
      dim_variance=dim_variance,
      norm_dim_variance=norm_dim_variance,
      is_ready=is_ready,
    )

  def decay_key(self, key: int | str) -> None:
    """Drop motion state when the entity is not visible this frame."""
    self._reset_key(key)


class HandDynamicHistory(_BBoxDynamicHistory):
  """Per-hand rolling bbox history (legacy API)."""

  def update(
    self,
    hand_side: str,
    hand_ltrb,
    dt_ns: int,
    timestamp_ns: int,
  ) -> HandMotionState:
    return super().update(hand_side, hand_ltrb, dt_ns, timestamp_ns)

  def decay_hand(self, hand_side: str) -> None:
    self.decay_key(hand_side)


class ObjectDynamicHistory(_BBoxDynamicHistory):
  """Per-object rolling bbox history keyed by HOT3D object_uid."""

  def update(
    self,
    object_uid: int,
    object_ltrb,
    dt_ns: int,
    timestamp_ns: int,
  ) -> MotionState:
    return super().update(object_uid, object_ltrb, dt_ns, timestamp_ns)

  def decay_object(self, object_uid: int) -> None:
    self.decay_key(object_uid)

  def decay_offscreen(self, visible_uids: set[int]) -> None:
    """Reset buffers for objects not visible in the current frame."""
    stale = [uid for uid in self._history if uid not in visible_uids]
    for uid in stale:
      self.decay_object(uid)
