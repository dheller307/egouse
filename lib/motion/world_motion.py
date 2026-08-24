"""World-frame 3D point velocity from HOT3D GT poses (m/s)."""

import numpy as np
from collections import deque
from dataclasses import dataclass

_EPS = 1e-9


@dataclass
class WorldMotionState:
  velocity: tuple[float, float, float] | None
  position: tuple[float, float, float] | None
  prev_position: tuple[float, float, float] | None
  is_ready: bool


def _translation_xyz(se3) -> tuple[float, float, float] | None:
  if se3 is None:
    return None
  t = np.asarray(se3.translation(), dtype=float).reshape(-1)
  if t.size < 3:
    return None
  return float(t[0]), float(t[1]), float(t[2])


def velocity_m_per_s(
  prev_xyz: tuple[float, float, float],
  curr_xyz: tuple[float, float, float],
  dt_ns: int,
) -> tuple[float, float, float] | None:
  if dt_ns <= 0:
    return None
  dt_s = dt_ns / 1e9
  return (
    (curr_xyz[0] - prev_xyz[0]) / dt_s,
    (curr_xyz[1] - prev_xyz[1]) / dt_s,
    (curr_xyz[2] - prev_xyz[2]) / dt_s,
  )


class WorldMotionHistory:
  """Rolling 3D world positions with smoothed center velocity (m/s)."""

  def __init__(
    self,
    max_history_len: int = 15,
    smooth_frames: int = 5,
    gap_reset_ms: float = 200.0,
  ):
    self._max_history_len = max_history_len
    self._smooth_frames = smooth_frames
    self._gap_reset_ms = gap_reset_ms
    self._history: dict[int | str, deque] = {}
    self._prev_smooth: dict[int | str, tuple[float, float, float]] = {}
    self._last_seen_ns: dict[int | str, int] = {}

  def _empty_state(self) -> WorldMotionState:
    return WorldMotionState(velocity=None, position=None, prev_position=None, is_ready=False)

  def _clear_buffer(self, key: int | str) -> None:
    self._prev_smooth.pop(key, None)
    if key in self._history:
      self._history[key].clear()

  def _reset_key(self, key: int | str) -> None:
    self._clear_buffer(key)
    self._last_seen_ns.pop(key, None)

  def update(
    self,
    key: int | str,
    xyz: tuple[float, float, float] | None,
    dt_ns: int,
    timestamp_ns: int,
  ) -> WorldMotionState:
    if xyz is None:
      self._reset_key(key)
      return self._empty_state()

    last_seen_ns = self._last_seen_ns.get(key)
    if last_seen_ns is not None:
      gap_ms = (timestamp_ns - last_seen_ns) / 1e6
      if gap_ms > self._gap_reset_ms:
        self._clear_buffer(key)

    if key not in self._history:
      self._history[key] = deque(maxlen=self._max_history_len)

    self._history[key].append(xyz)
    history_window = self._history[key]

    smooth_window = list(history_window)[-self._smooth_frames:]
    avg_x = sum(p[0] for p in smooth_window) / len(smooth_window)
    avg_y = sum(p[1] for p in smooth_window) / len(smooth_window)
    avg_z = sum(p[2] for p in smooth_window) / len(smooth_window)
    curr_smooth = (avg_x, avg_y, avg_z)

    is_ready = len(history_window) >= self._smooth_frames
    prev_smooth = self._prev_smooth.get(key)
    velocity = None
    if is_ready and prev_smooth is not None:
      velocity = velocity_m_per_s(prev_smooth, curr_smooth, dt_ns)
    self._prev_smooth[key] = curr_smooth
    self._last_seen_ns[key] = timestamp_ns

    return WorldMotionState(
      velocity=velocity,
      position=curr_smooth,
      prev_position=prev_smooth,
      is_ready=is_ready,
    )

  def decay_key(self, key: int | str) -> None:
    self._reset_key(key)

  def decay_offscreen(self, visible_keys: set[int | str]) -> None:
    stale = [k for k in self._history if k not in visible_keys]
    for key in stale:
      self.decay_key(key)


def se3_translation_xyz(se3) -> tuple[float, float, float] | None:
  return _translation_xyz(se3)
