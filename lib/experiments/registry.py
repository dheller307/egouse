"""Motion-frame registry for v2 dual-dwell experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

FrameScorer = Callable[..., list[dict]]


@dataclass(frozen=True)
class MotionFrameSpec:
  key: str
  label: str
  role: str
  score_frame: FrameScorer | None
  eval_label: str
  requires_mps: bool = False
  requires_imu: bool = False


def build_motion_frames() -> dict[str, MotionFrameSpec]:
  from lib.pipelines import gt_world

  return {
    "gt": MotionFrameSpec(
      key="gt",
      label="GT world (oracle)",
      role="Validates the dual-dwell decision layer using GT 3D wrist + object poses.",
      score_frame=gt_world.score_frame_v2_gt_world,
      eval_label="gt_world",
    ),
  }


_MOTION_FRAMES: dict[str, MotionFrameSpec] | None = None


def get_motion_frames() -> dict[str, MotionFrameSpec]:
  global _MOTION_FRAMES
  if _MOTION_FRAMES is None:
    _MOTION_FRAMES = build_motion_frames()
  return _MOTION_FRAMES


PRIMARY_FRAMES = ("gt",)


def parse_frame_keys(spec: str) -> list[str]:
  frames = get_motion_frames()
  keys = [k.strip().lower() for k in spec.split(",") if k.strip()]
  unknown = [k for k in keys if k not in frames]
  if unknown:
    raise ValueError(f"Unknown motion frame(s): {unknown}. Choose from {list(frames)}")
  return keys
