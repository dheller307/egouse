"""YOLOE open-vocabulary object detector (HOT3D text prompts)."""

from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLOE

from lib.detectors.backend import Detection, DetectorBackend
from lib.detectors.yoloe_prompts import load_hot3d_yoloe_prompts
from lib.paths import EGOUSE_ROOT

DEFAULT_YOLOE_WEIGHTS = EGOUSE_ROOT / "yoloe-26s-seg.pt"
DEFAULT_YOLOE_SIZE = "s"


def resolve_yoloe_weights(
  *,
  yoloe_weights: str | Path | None = None,
  yoloe_size: str | None = None,
) -> Path:
  if yoloe_weights is not None:
    return Path(yoloe_weights)
  env = os.environ.get("EGOUSE_YOLOE_WEIGHTS")
  if env:
    return Path(env)
  size = (yoloe_size or os.environ.get("EGOUSE_YOLOE_SIZE") or DEFAULT_YOLOE_SIZE).lower()
  name = f"yoloe-26{size}-seg.pt"
  local = EGOUSE_ROOT / name
  if local.is_file():
    return local
  return Path(name)


class YoloeBackend(DetectorBackend):
  """Ultralytics YOLOE with HOT3D manipulable text prompts (objects only)."""

  def __init__(
    self,
    *,
    yoloe_weights: str | Path | None = None,
    yoloe_size: str | None = None,
    conf: float = 0.25,
    device: str | None = None,
    prompts_path: Path | None = None,
  ) -> None:
    import torch

    if device is None:
      device = "cuda" if torch.cuda.is_available() else "cpu"
    self._device = device
    self._conf = conf
    self._yoloe_size = yoloe_size or os.environ.get("EGOUSE_YOLOE_SIZE") or DEFAULT_YOLOE_SIZE
    self._weights = resolve_yoloe_weights(yoloe_weights=yoloe_weights, yoloe_size=self._yoloe_size)

    vocab = load_hot3d_yoloe_prompts(prompts_path)
    self._prompt_list = vocab["prompt_list"]
    self._prompt_to_instance = vocab["prompt_to_instance"]

    self._model = YOLOE(str(self._weights))
    self._model.set_classes(self._prompt_list)

  @property
  def backend_name(self) -> str:
    return f"yoloe_26{self._yoloe_size}"

  @property
  def yoloe_size(self) -> str:
    return self._yoloe_size

  @property
  def yoloe_weights(self) -> Path:
    return self._weights

  @property
  def n_prompts(self) -> int:
    return len(self._prompt_list)

  @property
  def cuda(self) -> bool:
    return self._device.startswith("cuda")

  def infer(self, image_bgr) -> list[Detection]:
    results = self._model.predict(
      source=image_bgr,
      conf=self._conf,
      device=self._device,
      verbose=False,
    )
    out: list[Detection] = []
    if not results:
      return out
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
      return out

    names = results[0].names or {}
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss):
      prompt = str(names.get(int(cls_id), cls_id))
      instance_name = self._prompt_to_instance.get(prompt, prompt)
      out.append(
        Detection(
          label="obj",
          box=(float(x1), float(y1), float(x2), float(y2)),
          score=float(score),
          name=instance_name,
        )
      )
    return out

  def close(self) -> None:
    pass
