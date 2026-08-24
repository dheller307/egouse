"""Unit tests for dual-gate V2DwellTracker."""

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.scoring.interaction_v2 import V2DwellTracker

DT_NS = 33_300_000  # ~30 fps


class TestV2DwellTracker(unittest.TestCase):
  def setUp(self):
    self.tracker = V2DwellTracker(
      iou_dwell_ms=99.0,
      cm_dwell_ms=66.0,
      iou_gap_tolerance_ms=130.0,
      cm_gap_tolerance_ms=220.0,
    )

  def _step(self, iou_ok: bool, cm_ok: bool, oid: int = 1) -> None:
    self.tracker.update_hand_frame(
      "right",
      {oid: iou_ok},
      {oid: cm_ok},
      DT_NS,
    )

  def test_both_gates_required_to_interact(self):
    for _ in range(5):
      self._step(iou_ok=True, cm_ok=True)
    self.assertTrue(self.tracker.is_interacting("right", 1))

  def test_cm_pause_within_gap_preserves_cm_dwell(self):
    for _ in range(5):
      self._step(iou_ok=True, cm_ok=True)
    cm_before = self.tracker.get_cm_dwell_ms("right", 1)
    self.assertGreaterEqual(cm_before, 66.0)
    self._step(iou_ok=True, cm_ok=False)
    self.assertGreater(self.tracker.get_cm_dwell_ms("right", 1), 0.0)
    self.assertGreater(self.tracker.get_iou_dwell_ms("right", 1), cm_before)

  def test_iou_loss_resets_iou_dwell_faster_than_cm(self):
    for _ in range(5):
      self._step(iou_ok=True, cm_ok=True)
    for _ in range(5):
      self._step(iou_ok=False, cm_ok=False)
    self.assertEqual(self.tracker.get_iou_dwell_ms("right", 1), 0.0)

  def test_cm_requires_iou_to_accumulate(self):
    for _ in range(5):
      self._step(iou_ok=False, cm_ok=True)
    self.assertEqual(self.tracker.get_cm_dwell_ms("right", 1), 0.0)

  def test_keyboard_like_never_interacts_without_motion_ok(self):
    for _ in range(10):
      self._step(iou_ok=True, cm_ok=False)
    self.assertFalse(self.tracker.is_interacting("right", 1))

  def test_per_object_interacting_no_leader_gate(self):
    # Two objects both clearing dwell — both interact (no winner-take-all).
    for _ in range(5):
      self.tracker.update_hand_frame(
        "right",
        {1: True, 2: True},
        {1: True, 2: True},
        DT_NS,
      )
    self.assertTrue(self.tracker.is_object_interacting("right", 1))
    self.assertTrue(self.tracker.is_object_interacting("right", 2))
    self.assertEqual(set(self.tracker.interacting_object_ids("right")), {1, 2})

  def test_stationary_path_via_cm_ok_alias(self):
    # Stationary OR is applied upstream as motion_ok → cm_ok input here.
    for _ in range(5):
      self._step(iou_ok=True, cm_ok=True, oid=7)
    self.assertTrue(self.tracker.is_object_interacting("right", 7))


if __name__ == "__main__":
  unittest.main()
