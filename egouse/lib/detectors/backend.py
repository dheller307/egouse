"""Shared detector interface for Phase B student pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


Label = Literal["hand", "obj"]
HandSide = Literal["left", "right"]


@dataclass(frozen=True)
class Detection:
  """Single hand or object detection from RGB."""

  label: Label
  box: tuple[float, float, float, float]  # x1, y1, x2, y2
  score: float
  side: HandSide | None = None
  contact: int | None = None
  contact_score: float | None = None
  offset: tuple[float, float, float] | None = None  # magnitude, direction_x, direction_y
  raw_fields: tuple[float, ...] | None = None
  class_id: int | None = None
  prompt: str | None = None
  name: str | None = None  # optional detector class name (e.g. COCO)


class DetectorBackend(ABC):
  """Abstract detector: BGR image in, detections out."""

  @property
  @abstractmethod
  def backend_name(self) -> str:
    """Short tag for smoke_summary JSON."""

  @property
  @abstractmethod
  def cuda(self) -> bool:
    ...

  @abstractmethod
  def infer(self, image_bgr) -> list[Detection]:
    ...
