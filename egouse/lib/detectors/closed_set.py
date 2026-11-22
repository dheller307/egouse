"""Closed-set fine-tuned object detector: frozen class-ID -> HOT3D identity contract.

No text prompts and no post-hoc name mapping: the checkpoint's own class list must
equal the frozen vocabulary order exactly, or loading fails.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lib.detectors.backend import Detection, DetectorBackend

CONTRACT = "closed-set-v1"


def validate_class_mapping(names, vocabulary: list[str]) -> dict[int, str]:
  """Return {class_id: name} after strict checks against the frozen vocabulary order."""
  if len(set(vocabulary)) != len(vocabulary) or not vocabulary:
    raise ValueError("Frozen vocabulary must be non-empty and unique")
  if isinstance(names, dict):
    ids = list(names)
    if any(not isinstance(i, int) or isinstance(i, bool) for i in ids):
      raise ValueError("Checkpoint class IDs must be integers")
    mapping = {int(i): names[i] for i in ids}
  else:
    mapping = dict(enumerate(names))
  values = list(mapping.values())
  if any(not isinstance(v, str) for v in values):
    raise ValueError("Checkpoint class names must be strings")
  if len(set(values)) != len(values):
    raise ValueError("Duplicate class names in checkpoint")
  if sorted(mapping) != list(range(len(vocabulary))):
    raise ValueError(f"Checkpoint class IDs {sorted(mapping)} do not cover 0..{len(vocabulary) - 1} exactly")
  missing = sorted(set(vocabulary) - set(values))
  extra = sorted(set(values) - set(vocabulary))
  if missing or extra:
    raise ValueError(f"Incompatible classes: missing={missing} extra={extra}")
  wrong = [i for i, name in enumerate(vocabulary) if mapping[i] != name]
  if wrong:
    raise ValueError(f"Class order differs from frozen vocabulary at IDs {wrong}")
  return mapping


def load_detector_config(path: Path) -> dict:
  config = json.loads(Path(path).read_text())
  required = {"contract", "checkpoint", "checkpoint_sha256", "vocabulary", "conf", "imgsz"}
  if not required <= set(config):
    raise ValueError(f"Detector config missing {sorted(required - set(config))}")
  if config["contract"] != CONTRACT:
    raise ValueError(f"Unsupported detector contract {config['contract']!r}")
  return config


class ClosedSetYoloBackend(DetectorBackend):
  """Fine-tuned Ultralytics detection checkpoint whose head *is* the HOT3D vocabulary."""

  def __init__(self, config: dict, *, device: str = "cuda:0") -> None:
    from ultralytics import YOLO

    checkpoint = Path(config["checkpoint"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != config["checkpoint_sha256"]:
      raise ValueError("Detector checkpoint content differs from the frozen config")
    self._model = YOLO(str(checkpoint))
    if self._model.task != "detect":
      raise ValueError(f"Closed-set contract needs a detection checkpoint, got task={self._model.task!r}")
    self._names = validate_class_mapping(self._model.names, list(config["vocabulary"]))
    self._conf = float(config["conf"])
    self._imgsz = int(config["imgsz"])
    self._device = device
    self.checkpoint_sha256 = digest

  @property
  def backend_name(self) -> str:
    return CONTRACT

  @property
  def cuda(self) -> bool:
    return self._device.startswith("cuda")

  def infer(self, image_bgr) -> list[Detection]:
    result = self._model.predict(source=image_bgr, conf=self._conf, imgsz=self._imgsz, device=self._device, verbose=False)[0]
    out = []
    for (x1, y1, x2, y2), score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                           result.boxes.cls.cpu().int().tolist()):
      if cls not in self._names:
        raise ValueError(f"Detector emitted unknown class ID {cls}")
      out.append(Detection(label="obj", box=(float(x1), float(y1), float(x2), float(y2)), score=float(score),
                           class_id=int(cls), name=self._names[cls], prompt=None))
    return out


def detection_record(d: Detection) -> dict:
  """Serialized object detection; `prompt` is absent by contract."""
  return {"box": list(d.box), "score": d.score, "name": d.name, "class_id": d.class_id}


def validate_object_records(objects: list[dict], vocabulary: list[str]) -> None:
  for o in objects:
    if "prompt" in o:
      raise ValueError("Closed-set object records must not carry text prompts")
    cid = o.get("class_id")
    if not isinstance(cid, int) or not 0 <= cid < len(vocabulary) or vocabulary[cid] != o.get("name"):
      raise ValueError(f"Object record violates class-ID mapping: {o}")
    if len(o["box"]) != 4 or not 0 <= o["score"] <= 1:
      raise ValueError(f"Invalid object box/score: {o}")
