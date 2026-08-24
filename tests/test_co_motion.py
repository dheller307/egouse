"""Unit tests for co-motion scoring."""

import unittest

from lib.scoring.co_motion import (
  co_motion_ok,
  co_motion_score,
  co_motion_score_xyz,
  speed,
  stationary_ok,
  world_speed,
)


class TestCoMotion(unittest.TestCase):
  def test_same_direction_full_alignment(self):
    hand_vel = (100.0, 0.0)
    object_vel = (50.0, 0.0)
    score = co_motion_score(hand_vel, object_vel, v_min=10.0)
    self.assertAlmostEqual(score, 1.0)

  def test_opposite_direction_zero(self):
    hand_vel = (100.0, 0.0)
    object_vel = (-80.0, 0.0)
    score = co_motion_score(hand_vel, object_vel, v_min=10.0)
    self.assertAlmostEqual(score, 0.0)

  def test_orthogonal_direction_zero(self):
    hand_vel = (100.0, 0.0)
    object_vel = (0.0, 60.0)
    score = co_motion_score(hand_vel, object_vel, v_min=10.0)
    self.assertAlmostEqual(score, 0.0)

  def test_below_v_min_returns_zero(self):
    hand_vel = (5.0, 0.0)
    object_vel = (100.0, 0.0)
    self.assertEqual(co_motion_score(hand_vel, object_vel, v_min=30.0), 0.0)
    self.assertEqual(co_motion_score((100.0, 0.0), (5.0, 0.0), v_min=30.0), 0.0)

  def test_none_velocity_returns_zero(self):
    self.assertEqual(co_motion_score(None, (10.0, 0.0), v_min=1.0), 0.0)
    self.assertEqual(co_motion_score((10.0, 0.0), None, v_min=1.0), 0.0)

  def test_partial_alignment(self):
    hand_vel = (100.0, 0.0)
    object_vel = (100.0, 100.0)
    score = co_motion_score(hand_vel, object_vel, v_min=10.0)
    self.assertAlmostEqual(score, 0.70710678, places=5)

  def test_co_motion_ok_threshold(self):
    self.assertTrue(co_motion_ok(0.6, 0.5))
    self.assertFalse(co_motion_ok(0.4, 0.5))

  def test_speed_helper(self):
    self.assertAlmostEqual(speed((3.0, 4.0)), 5.0)
    self.assertEqual(speed(None), 0.0)

  def test_world_speed_3d(self):
    self.assertAlmostEqual(world_speed((0.0, 0.0, 0.03)), 0.03)
    self.assertEqual(world_speed(None), 0.0)

  def test_xyz_co_motion_vertical_lift(self):
    # XY-only would return 0 (both below XY v_min); XYZ should align.
    hand_vel = (0.0, 0.0, 0.05)
    object_vel = (0.0, 0.0, 0.04)
    self.assertEqual(co_motion_score(hand_vel[:2], object_vel[:2], v_min=0.01), 0.0)
    self.assertAlmostEqual(co_motion_score_xyz(hand_vel, object_vel, v_min=0.01), 1.0)

  def test_stationary_ok_aabb_no_iou(self):
    self.assertTrue(
      stationary_ok(
        (0.005, 0.0, 0.0),
        (0.001, 0.0, 0.0),
        v_max_m_s=0.015,
        aabb_surface_dist_m=0.02,
        aabb_d_max_m=0.05,
      )
    )

  def test_stationary_rejects_far_or_fast(self):
    self.assertFalse(
      stationary_ok(
        (0.05, 0.0, 0.0),
        (0.001, 0.0, 0.0),
        v_max_m_s=0.015,
        aabb_surface_dist_m=0.02,
        aabb_d_max_m=0.05,
      )
    )
    self.assertFalse(
      stationary_ok(
        (0.005, 0.0, 0.0),
        (0.001, 0.0, 0.0),
        v_max_m_s=0.015,
        aabb_surface_dist_m=0.08,
        aabb_d_max_m=0.05,
      )
    )
    self.assertFalse(
      stationary_ok(
        None,
        (0.0, 0.0, 0.0),
        v_max_m_s=0.015,
        aabb_surface_dist_m=0.01,
        aabb_d_max_m=0.05,
      )
    )


class TestObjectAabb(unittest.TestCase):
  def test_point_inside_and_outside(self):
    import numpy as np
    from lib.motion.object_aabb import point_to_aabb_distance

    mn = np.array([-0.1, -0.05, -0.02])
    mx = np.array([0.1, 0.05, 0.02])
    self.assertEqual(point_to_aabb_distance((0.0, 0.0, 0.0), mn, mx), 0.0)
    self.assertAlmostEqual(point_to_aabb_distance((0.0, 0.0, 0.05), mn, mx), 0.03)


if __name__ == "__main__":
  unittest.main()
