"""Object-local AABB from HOT3D CAD GLBs + wrist-to-surface distance."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

# HOT3D CAD meshes store vertex positions in millimeters; poses are meters.
_GLB_UNIT_TO_METERS = 0.001


def point_to_aabb_distance(
  point: tuple[float, float, float] | np.ndarray,
  aabb_min: np.ndarray,
  aabb_max: np.ndarray,
) -> float:
  """Euclidean distance from a point to an axis-aligned box (0 if inside)."""
  p = np.asarray(point, dtype=float).reshape(3)
  closest = np.minimum(np.maximum(p, aabb_min), aabb_max)
  return float(np.linalg.norm(p - closest))


def transform_world_point_to_object(
  point_world: tuple[float, float, float],
  T_world_object,
) -> np.ndarray:
  """Map a world point into the object frame using T_world_object (object→world)."""
  R = np.asarray(T_world_object.rotation().to_matrix(), dtype=float).reshape(3, 3)
  t = np.asarray(T_world_object.translation(), dtype=float).reshape(3)
  p = np.asarray(point_world, dtype=float).reshape(3)
  return R.T @ (p - t)


def glb_local_aabb_m(glb_path: Path) -> tuple[np.ndarray, np.ndarray]:
  """
  Axis-aligned bounds of POSITION accessors in a GLB, converted to meters.

  Uses accessor min/max when present (standard for HOT3D assets).
  """
  data = Path(glb_path).read_bytes()
  magic, _version, _length = struct.unpack_from("<4sII", data, 0)
  if magic != b"glTF":
    raise ValueError(f"Not a GLB file: {glb_path}")
  chunk0_len, _chunk0_type = struct.unpack_from("<I4s", data, 12)
  gltf = json.loads(data[20 : 20 + chunk0_len])

  mins: list[list[float]] = []
  maxs: list[list[float]] = []
  for mesh in gltf.get("meshes", []):
    for prim in mesh.get("primitives", []):
      attrs = prim.get("attributes") or {}
      if "POSITION" not in attrs:
        continue
      acc = gltf["accessors"][attrs["POSITION"]]
      if "min" not in acc or "max" not in acc:
        raise ValueError(f"GLB accessor missing min/max: {glb_path}")
      mins.append(acc["min"])
      maxs.append(acc["max"])
  if not mins:
    raise ValueError(f"No POSITION accessors in {glb_path}")

  aabb_min = np.min(np.asarray(mins, dtype=float), axis=0) * _GLB_UNIT_TO_METERS
  aabb_max = np.max(np.asarray(maxs, dtype=float), axis=0) * _GLB_UNIT_TO_METERS
  return aabb_min, aabb_max


class ObjectAabbLibrary:
  """Cached object-local AABBs (meters) loaded from HOT3D CAD GLBs."""

  def __init__(self, asset_folder: str | Path):
    self._asset_folder = Path(asset_folder)
    self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

  def get(self, object_uid: str | int) -> tuple[np.ndarray, np.ndarray] | None:
    key = str(object_uid)
    if key in self._cache:
      return self._cache[key]
    glb_path = self._asset_folder / f"{key}.glb"
    if not glb_path.is_file():
      return None
    try:
      aabb = glb_local_aabb_m(glb_path)
    except (ValueError, OSError, json.JSONDecodeError, struct.error):
      return None
    self._cache[key] = aabb
    return aabb

  def surface_distance_m(
    self,
    object_uid: str | int,
    wrist_xyz_world: tuple[float, float, float],
    T_world_object,
  ) -> float | None:
    """Wrist-to-object AABB surface distance in meters, or None if unavailable."""
    aabb = self.get(object_uid)
    if aabb is None or T_world_object is None:
      return None
    aabb_min, aabb_max = aabb
    p_local = transform_world_point_to_object(wrist_xyz_world, T_world_object)
    return point_to_aabb_distance(p_local, aabb_min, aabb_max)
