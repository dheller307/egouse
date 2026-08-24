"""Unit tests for co-motion proximity gate (IoU vs AABB)."""

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.scoring.co_motion import co_motion_proximity_ok


class TestCoMotionProximity(unittest.TestCase):
  def test_frozen_requires_iou(self):
    # AABB near + iou fail → prox false when flag off.
    self.assertFalse(
      co_motion_proximity_ok(
        iou_ok=False,
        aabb_surface_dist_m=0.05,
        aabb_d_max_m=0.12,
        enable_aabb_co_motion=False,
      )
    )
    self.assertTrue(
      co_motion_proximity_ok(
        iou_ok=True,
        aabb_surface_dist_m=0.05,
        aabb_d_max_m=0.12,
        enable_aabb_co_motion=False,
      )
    )

  def test_aabb_flag_uses_surface_distance(self):
    # AABB near + iou fail → prox true when flag on.
    self.assertTrue(
      co_motion_proximity_ok(
        iou_ok=False,
        aabb_surface_dist_m=0.05,
        aabb_d_max_m=0.12,
        enable_aabb_co_motion=True,
      )
    )
    self.assertFalse(
      co_motion_proximity_ok(
        iou_ok=False,
        aabb_surface_dist_m=0.20,
        aabb_d_max_m=0.12,
        enable_aabb_co_motion=True,
      )
    )

  def test_aabb_flag_rejects_missing_dist(self):
    self.assertFalse(
      co_motion_proximity_ok(
        iou_ok=True,
        aabb_surface_dist_m=None,
        aabb_d_max_m=0.12,
        enable_aabb_co_motion=True,
      )
    )


if __name__ == "__main__":
  unittest.main()
