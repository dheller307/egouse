"""Detector boundary contracts, independent of Torch and vendor code."""
from __future__ import annotations

import math
from lib.detectors.backend import Detection

HAND_SCHEMA = "100doh-row-v2"
OFFSET_FIELDS = ("magnitude", "direction_x", "direction_y")


def parse_doh_hand_row(row) -> Detection:
  """Upstream: xyxy, score, contact, magnitude, dx, dy, right-hand bit."""
  values = tuple(float(x) for x in row)
  if len(values) != 10 or not all(math.isfinite(x) for x in values):
    raise ValueError("100DOH hand row must contain exactly 10 finite values")
  if values[5] not in range(5) or values[9] not in (0., 1.):
    raise ValueError("invalid 100DOH contact state or handedness bit")
  if not 0 <= values[4] <= 1 or values[2] < values[0] or values[3] < values[1]:
    raise ValueError("invalid 100DOH score or box")
  return Detection(label="hand", box=values[:4], score=values[4],
                   contact=int(values[5]), side="right" if values[9] == 1 else "left",
                   offset=values[6:9], raw_fields=values)


def best_hand_for_side(hands: list[dict], side: str) -> dict | None:
  """Missing/unknown side remains missing; never borrow the opposite hand."""
  if side not in ("left", "right"):
    raise ValueError(f"unknown requested hand side: {side}")
  candidates = [h for h in hands if h.get("side") == side
                and math.isfinite(float(h.get("score") or 0.))]
  return max(candidates, key=lambda h: float(h.get("score") or 0.)) if candidates else None
