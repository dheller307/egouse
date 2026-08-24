"""v2 interaction: dual-gate (IoU + co-motion) dwell with separate gaps."""

from dataclasses import dataclass


@dataclass
class V2FrameResult:
  iou_dwell_ms_by_object: dict[int, float]
  cm_dwell_ms_by_object: dict[int, float]
  dwell_ms_by_object: dict[int, float]
  leader_id: int | None


class V2DwellTracker:
  """
  Per (hand, object) IoU and co-motion dwell clocks with independent gap decay.

  - iou_dwell accumulates while iou_ok
  - cm_dwell accumulates while iou_ok and co_motion_ok
  - An object is interacting when both clocks exceed thresholds (per-object;
    no winner-take-all). Frame-level export may still pick a leader for display.
  """

  def __init__(
    self,
    iou_dwell_ms: float = 99.0,
    cm_dwell_ms: float = 66.0,
    iou_gap_tolerance_ms: float = 130.0,
    cm_gap_tolerance_ms: float = 220.0,
  ):
    self.iou_dwell_ms = iou_dwell_ms
    self.cm_dwell_ms = cm_dwell_ms
    self.iou_gap_tolerance_ms = iou_gap_tolerance_ms
    self.cm_gap_tolerance_ms = cm_gap_tolerance_ms
    self._iou_dwell: dict[tuple[str, int], float] = {}
    self._cm_dwell: dict[tuple[str, int], float] = {}
    self._iou_missed: dict[tuple[str, int], float] = {}
    self._cm_missed: dict[tuple[str, int], float] = {}

  def _key(self, hand_side: str, object_id: int) -> tuple[str, int]:
    return (hand_side, object_id)

  def _tracked_object_ids(self, hand_side: str) -> set[int]:
    ids: set[int] = set()
    for side, object_id in (
      *self._iou_dwell.keys(),
      *self._cm_dwell.keys(),
      *self._iou_missed.keys(),
      *self._cm_missed.keys(),
    ):
      if side == hand_side:
        ids.add(object_id)
    return ids

  def _accumulate_iou(self, hand_side: str, object_id: int, dt_ms: float) -> None:
    key = self._key(hand_side, object_id)
    self._iou_dwell[key] = self._iou_dwell.get(key, 0.0) + dt_ms
    self._iou_missed[key] = 0.0

  def _accumulate_cm(self, hand_side: str, object_id: int, dt_ms: float) -> None:
    key = self._key(hand_side, object_id)
    self._cm_dwell[key] = self._cm_dwell.get(key, 0.0) + dt_ms
    self._cm_missed[key] = 0.0

  def _gap_decay_iou(self, hand_side: str, object_id: int, dt_ms: float) -> None:
    key = self._key(hand_side, object_id)
    self._iou_missed[key] = self._iou_missed.get(key, 0.0) + dt_ms
    if self._iou_missed[key] >= self.iou_gap_tolerance_ms:
      self._iou_dwell[key] = 0.0
      self._iou_missed[key] = 0.0

  def _gap_decay_cm(self, hand_side: str, object_id: int, dt_ms: float) -> None:
    key = self._key(hand_side, object_id)
    self._cm_missed[key] = self._cm_missed.get(key, 0.0) + dt_ms
    if self._cm_missed[key] >= self.cm_gap_tolerance_ms:
      self._cm_dwell[key] = 0.0
      self._cm_missed[key] = 0.0

  def _gap_decay_offscreen(
    self,
    hand_side: str,
    visible_object_ids: set[int],
    dt_ms: float,
  ) -> None:
    for object_id in self._tracked_object_ids(hand_side):
      if object_id not in visible_object_ids:
        self._gap_decay_iou(hand_side, object_id, dt_ms)
        self._gap_decay_cm(hand_side, object_id, dt_ms)

  def _bottleneck_dwell(self, hand_side: str, object_id: int) -> float:
    return min(
      self.get_iou_dwell_ms(hand_side, object_id),
      self.get_cm_dwell_ms(hand_side, object_id),
    )

  def update_hand_frame(
    self,
    hand_side: str,
    iou_ok_by_object: dict[int, bool],
    cm_ok_by_object: dict[int, bool],
    dt_ns: int,
  ) -> V2FrameResult:
    dt_ms = dt_ns / 1e6
    visible_object_ids = set(iou_ok_by_object.keys())

    if not iou_ok_by_object:
      self._gap_decay_offscreen(hand_side, visible_object_ids, dt_ms)
      return V2FrameResult(
        iou_dwell_ms_by_object={},
        cm_dwell_ms_by_object={},
        dwell_ms_by_object={},
        leader_id=None,
      )

    for object_id in iou_ok_by_object:
      iou_ok = iou_ok_by_object[object_id]
      cm_ok = cm_ok_by_object.get(object_id, False)
      if iou_ok:
        self._accumulate_iou(hand_side, object_id, dt_ms)
      else:
        self._gap_decay_iou(hand_side, object_id, dt_ms)
      if iou_ok and cm_ok:
        self._accumulate_cm(hand_side, object_id, dt_ms)
      else:
        self._gap_decay_cm(hand_side, object_id, dt_ms)

    self._gap_decay_offscreen(hand_side, visible_object_ids, dt_ms)

    iou_dwell_ms_by_object = {
      object_id: self.get_iou_dwell_ms(hand_side, object_id)
      for object_id in iou_ok_by_object
    }
    cm_dwell_ms_by_object = {
      object_id: self.get_cm_dwell_ms(hand_side, object_id)
      for object_id in iou_ok_by_object
    }
    dwell_ms_by_object = {
      object_id: min(
        iou_dwell_ms_by_object[object_id],
        cm_dwell_ms_by_object[object_id],
      )
      for object_id in iou_ok_by_object
    }
    leader_id = self._pick_leader(hand_side, dwell_ms_by_object)

    return V2FrameResult(
      iou_dwell_ms_by_object=iou_dwell_ms_by_object,
      cm_dwell_ms_by_object=cm_dwell_ms_by_object,
      dwell_ms_by_object=dwell_ms_by_object,
      leader_id=leader_id,
    )

  def get_iou_dwell_ms(self, hand_side: str, object_id: int) -> float:
    return self._iou_dwell.get(self._key(hand_side, object_id), 0.0)

  def get_cm_dwell_ms(self, hand_side: str, object_id: int) -> float:
    return self._cm_dwell.get(self._key(hand_side, object_id), 0.0)

  def get_dwell_ms(self, hand_side: str, object_id: int) -> float:
    """Bottleneck dwell min(iou, cm) for timeline / backward compatibility."""
    return self._bottleneck_dwell(hand_side, object_id)

  def decay_hand(self, hand_side: str, dt_ns: int) -> None:
    dt_ms = dt_ns / 1e6
    for object_id in self._tracked_object_ids(hand_side):
      self._gap_decay_iou(hand_side, object_id, dt_ms)
      self._gap_decay_cm(hand_side, object_id, dt_ms)

  def _pick_leader(
    self,
    hand_side: str,
    dwell_ms_by_object: dict[int, float],
  ) -> int | None:
    if not dwell_ms_by_object:
      return None
    return max(dwell_ms_by_object, key=lambda oid: (dwell_ms_by_object[oid], oid))

  def best_object(
    self,
    hand_side: str,
    frame_iou: dict[int, float] | None = None,
    candidate_ids: set[int] | None = None,
  ) -> int | None:
    """Leader = max bottleneck dwell; tie-break on frame IoU."""
    best_dwell = 0.0
    best_id = None
    for object_id in self._tracked_object_ids(hand_side):
      if candidate_ids is not None and object_id not in candidate_ids:
        continue
      dwell = self._bottleneck_dwell(hand_side, object_id)
      if dwell > best_dwell:
        best_dwell = dwell
        best_id = object_id
      elif dwell == best_dwell and dwell > 0:
        if best_id is None:
          best_id = object_id
        elif frame_iou is not None and frame_iou.get(object_id, 0) > frame_iou.get(best_id, 0):
          best_id = object_id
    return best_id

  def is_object_interacting(self, hand_side: str, object_id: int | None) -> bool:
    """True when this (hand, object) pair clears both dwell thresholds."""
    if object_id is None:
      return False
    if self.get_iou_dwell_ms(hand_side, object_id) < self.iou_dwell_ms:
      return False
    if self.get_cm_dwell_ms(hand_side, object_id) < self.cm_dwell_ms:
      return False
    return True

  def interacting_object_ids(self, hand_side: str) -> list[int]:
    return [
      object_id
      for object_id in self._tracked_object_ids(hand_side)
      if self.is_object_interacting(hand_side, object_id)
    ]

  def is_interacting(self, hand_side: str, object_id: int | None) -> bool:
    """Backward-compatible alias for per-object interacting (no leader gate)."""
    return self.is_object_interacting(hand_side, object_id)

  def pick_export_leader(
    self,
    hand_side: str,
    frame_iou: dict[int, float] | None = None,
    candidate_ids: set[int] | None = None,
  ) -> int | None:
    """Prefer an interacting object; else fall back to best dwell."""
    interacting = self.interacting_object_ids(hand_side)
    if candidate_ids is not None:
      interacting = [oid for oid in interacting if oid in candidate_ids]
    if interacting:
      iou = frame_iou or {}
      return max(
        interacting,
        key=lambda oid: (self._bottleneck_dwell(hand_side, oid), iou.get(oid, 0.0), oid),
      )
    return self.best_object(hand_side, frame_iou, candidate_ids)
