"""Shared student feature column schema (v2: 15 numeric detector features)."""

from __future__ import annotations

FEATURE_SCHEMA = "hybrid-v2-offset3-strict-side-seconds"

FEATURE_COLUMNS = [
  "hand_obj_iou",
  "hand_obj_center_dist_norm",
  "area_ratio",
  "hand_score",
  "hand_contact",
  "hand_offset_magnitude",
  "hand_offset_direction_x",
  "hand_offset_direction_y",
  "obj_score",
  "rel_speed",
  "iou_roll_mean",
  "dist_roll_mean",
  "dist_roll_std",
  "dist_roll_slope",
  "obj_present_frac",
]

ROLL_K = 15
ROLL_MIN_VALID = 3
