"""Shared student feature column schema (14-D hybrid detector features)."""

from __future__ import annotations

FEATURE_COLUMNS = [
  "hand_obj_iou",
  "hand_obj_center_dist_norm",
  "area_ratio",
  "hand_score",
  "hand_contact",
  "hand_offset_x",
  "hand_offset_y",
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
