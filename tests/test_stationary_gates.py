"""Unit tests for stationary asymmetric speeds and separate AABB-cm radius."""

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.scoring.co_motion import co_motion_proximity_ok, stationary_ok, world_speed


class TestStationaryAsymmetric(unittest.TestCase):
  def _vel(self, speed: float) -> tuple[float, float, float]:
    return (speed, 0.0, 0.0)

  def test_symmetric_default(self):
    hand = self._vel(0.01)
    obj = self._vel(0.01)
    self.assertTrue(
      stationary_ok(
        hand, obj, v_max_m_s=0.015, aabb_surface_dist_m=0.05, aabb_d_max_m=0.12
      )
    )
    self.assertFalse(
      stationary_ok(
        hand, obj, v_max_m_s=0.015, aabb_surface_dist_m=0.05, aabb_d_max_m=0.12,
        hand_v_max_m_s=0.008,
      )
    )

  def test_looser_hand_only(self):
    hand = self._vel(0.03)
    obj = self._vel(0.01)
    self.assertFalse(
      stationary_ok(
        hand, obj, v_max_m_s=0.015, aabb_surface_dist_m=0.05, aabb_d_max_m=0.12
      )
    )
    self.assertTrue(
      stationary_ok(
        hand, obj, v_max_m_s=0.015, aabb_surface_dist_m=0.05, aabb_d_max_m=0.12,
        hand_v_max_m_s=0.04,
      )
    )

  def test_object_still_anchors(self):
    hand = self._vel(0.01)
    obj = self._vel(0.02)
    self.assertFalse(
      stationary_ok(
        hand, obj, v_max_m_s=0.015, aabb_surface_dist_m=0.05, aabb_d_max_m=0.12,
        hand_v_max_m_s=0.04,
      )
    )


class TestSeparateCoMotionAabb(unittest.TestCase):
  def test_different_radius_for_aabb_cm(self):
    # Between 0.10 and 0.16: passes cm radius, would fail stationary-tight 0.10
    self.assertTrue(
      co_motion_proximity_ok(
        iou_ok=False,
        aabb_surface_dist_m=0.12,
        aabb_d_max_m=0.16,
        enable_aabb_co_motion=True,
      )
    )
    self.assertFalse(
      co_motion_proximity_ok(
        iou_ok=False,
        aabb_surface_dist_m=0.12,
        aabb_d_max_m=0.10,
        enable_aabb_co_motion=True,
      )
    )


if __name__ == "__main__":
  unittest.main()
