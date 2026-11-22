from dataclasses import dataclass
import numpy as np


@dataclass
class Box2d:
  left: float
  top: float
  right: float
  bottom: float


def orient_bbox_cw(bbox, image_height) -> Box2d | None:
  if bbox is None:
    return None
  corners = np.array(
    [
      [bbox.left, bbox.top],
      [bbox.right, bbox.top],
      [bbox.right, bbox.bottom],
      [bbox.left, bbox.bottom],
    ],
    dtype=float,
  )
  rotated = np.stack([image_height - 1 - corners[:, 1], corners[:, 0]], axis=1)
  return Box2d(
    left=float(rotated[:, 0].min()),
    top=float(rotated[:, 1].min()),
    right=float(rotated[:, 0].max()),
    bottom=float(rotated[:, 1].max()),
  )


def bbox_to_ltrb(bbox, image_height, rotate=True) -> tuple[float, float, float, float] | None:
  if bbox is None:
    return None
  if rotate:
    oriented = orient_bbox_cw(bbox, image_height)
    if oriented is None:
      return None
    return (oriented.left, oriented.top, oriented.right, oriented.bottom)
  return (bbox.left, bbox.top, bbox.right, bbox.bottom)


def bbox_center(ltrb) -> tuple[float, float] | None:
  if ltrb is None:
    return None
  bl, bt, br, bb = ltrb
  return (bl + br) / 2, (bt + bb) / 2


def hand_velocity_px_per_s(prev_center, curr_center, dt_ns) -> tuple[float, float] | None:
  """Velocity (vx, vy) in pixels per second."""
  if prev_center is None or curr_center is None or dt_ns <= 0:
    return None
  dt_s = dt_ns / 1e9
  prev_x, prev_y = prev_center
  curr_x, curr_y = curr_center
  vx = (curr_x - prev_x) / dt_s
  vy = (curr_y - prev_y) / dt_s
  return vx, vy


def unit_vector(ax, ay) -> tuple[float, float] | None:
  """Unit vector conversion"""
  norm = np.sqrt(ax ** 2 + ay ** 2)
  if norm <= 1e-6:
    return None
  return ax / norm, ay / norm


def approach_alignment(velocity, hand_ltrb, object_ltrb) -> float:
  """cos(angle) between hand velocity and vector hand_center->object_center"""
  """Returns value in [0, 1] (mapped from [-1, 1]), fallback of 0.5 when velocity is None or near 0."""
  if velocity is None:
    return 0.5
  u_velocity = unit_vector(*velocity)
  if u_velocity is None:
    return 0.5

  hand_center = bbox_center(hand_ltrb)
  object_center = bbox_center(object_ltrb)
  if hand_center is None or object_center is None:
    return 0.5
  u_hand_to_object = unit_vector(object_center[0] - hand_center[0], object_center[1] - hand_center[1])
  if u_hand_to_object is None:
    return 0.5
  dot_product = np.dot(u_velocity, u_hand_to_object)
  return max(0.0, min(1.0, dot_product))


def engagement_confidence(spatial, approach_score, w_spatial=0.7, w_approach=0.3) -> float:
  """Weighted sum of spatial and approach scores."""
  """Instantaneous engagement confidence."""
  return w_spatial * spatial + w_approach * approach_score

def bbox_iou(box_a, box_b) -> float | None:
  if box_a is None or box_b is None:
    return 0.0
  al, at, ar, ab = box_a
  bl, bt, br, bb = box_b
  inter_w = max(0.0, min(ar, br) - max(al, bl))
  inter_h = max(0.0, min(ab, bb) - max(at, bt))
  intersection_area = inter_w * inter_h
  union_area = (ar - al) * (ab - at) + (br - bl) * (bb - bt) - intersection_area
  return intersection_area / union_area if union_area > 0 else 0.0


def center_distance(box_a, box_b) -> float | None:
  if box_a is None or box_b is None:
    return None
  cx_a, cy_a = bbox_center(box_a)
  cx_b, cy_b = bbox_center(box_b)
  return float(np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2))


def norm_center_distance(box_a, box_b) -> float | None:
  """Center distance normalized by sqrt(object area)."""
  dist = center_distance(box_a, box_b)
  if dist is None:
    return None
  bl, bt, br, bb = box_b
  object_area = (br - bl) * (bb - bt)
  if object_area <= 0:
    return None
  return dist / float(np.sqrt(object_area))


def distance_confidence(box_a, box_b, k) -> float:
  """Confidence in [0, 1]; higher when hand center is near object center."""
  if box_a is None or box_b is None or k <= 0:
    return 0.0
  d_norm = norm_center_distance(box_a, box_b)
  if d_norm is None:
    return 0.0
  return float(max(0.0, 1.0 - d_norm / k))


def spatial_confidence(iou_score, dist_score, w_iou=0.5, w_dist=0.50) -> float:
  """Weighted sum of iou and distance scores."""
  return w_iou * iou_score + w_dist * dist_score
