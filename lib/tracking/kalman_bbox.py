"""Optional OpenCV Kalman filter for 2D object box tracks (center + size).

Not used by the student or teacher pipelines. `eval_hybrid_boxes.py --no-kalman`
is the default student path (raw YOLOE boxes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from lib.geometry import bbox_iou


def _ltrb_to_cxcywh(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
  x1, y1, x2, y2 = box
  w = max(1.0, x2 - x1)
  h = max(1.0, y2 - y1)
  return (x1 + w / 2, y1 + h / 2, w, h)


def _cxcywh_to_ltrb(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
  return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


@dataclass
class TrackedBox:
  track_id: int
  box: tuple[float, float, float, float]
  score: float
  name: str | None
  age: int
  hits: int
  coasting: bool
  time_since_update: int


@dataclass
class KalmanBBoxTracker:
  """Greedy IoU assignment + per-track Kalman on (cx, cy, w, h)."""

  iou_thresh: float = 0.3
  max_age: int = 20
  min_hits: int = 1
  _next_id: int = field(default=1, init=False)
  _tracks: list[dict] = field(default_factory=list, init=False)

  def _new_kf(self, cx: float, cy: float, w: float, h: float) -> cv2.KalmanFilter:
    # State: [cx, cy, w, h, dcx, dcy, dw, dh]
    kf = cv2.KalmanFilter(8, 4)
    kf.transitionMatrix = np.eye(8, dtype=np.float32)
    for i in range(4):
      kf.transitionMatrix[i, i + 4] = 1.0
    kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
    for i in range(4):
      kf.measurementMatrix[i, i] = 1.0
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
    kf.errorCovPost = np.eye(8, dtype=np.float32)
    kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
    return kf

  def update(
    self,
    detections: list[dict],
  ) -> list[TrackedBox]:
    """Update with detections: each dict has box, score, optional name."""
    if not self._tracks and not detections:
      return []

    # Predict all tracks
    for tr in self._tracks:
      pred = tr["kf"].predict()
      tr["pred_box"] = _cxcywh_to_ltrb(
        float(pred[0, 0]), float(pred[1, 0]), float(pred[2, 0]), float(pred[3, 0])
      )
      tr["time_since_update"] += 1

    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()

    if self._tracks and detections:
      pairs: list[tuple[float, int, int]] = []
      for ti, tr in enumerate(self._tracks):
        for di, det in enumerate(detections):
          iou = float(bbox_iou(tr["pred_box"], det["box"]) or 0.0)
          if iou >= self.iou_thresh:
            pairs.append((iou, ti, di))
      pairs.sort(reverse=True)
      for _, ti, di in pairs:
        if ti in matched_tracks or di in matched_dets:
          continue
        matched_tracks.add(ti)
        matched_dets.add(di)
        tr = self._tracks[ti]
        det = detections[di]
        cx, cy, w, h = _ltrb_to_cxcywh(det["box"])
        meas = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        tr["kf"].correct(meas)
        st = tr["kf"].statePost
        tr["box"] = _cxcywh_to_ltrb(float(st[0, 0]), float(st[1, 0]), float(st[2, 0]), float(st[3, 0]))
        tr["score"] = float(det.get("score", tr["score"]))
        if det.get("name"):
          tr["name"] = det["name"]
        tr["hits"] += 1
        tr["time_since_update"] = 0
        tr["age"] += 1

    # Unmatched detections -> new tracks
    for di, det in enumerate(detections):
      if di in matched_dets:
        continue
      cx, cy, w, h = _ltrb_to_cxcywh(det["box"])
      self._tracks.append(
        {
          "id": self._next_id,
          "kf": self._new_kf(cx, cy, w, h),
          "box": det["box"],
          "score": float(det.get("score", 0.0)),
          "name": det.get("name"),
          "hits": 1,
          "time_since_update": 0,
          "age": 1,
          "pred_box": det["box"],
        }
      )
      self._next_id += 1

    # Coast unmatched tracks (use prediction)
    for ti, tr in enumerate(self._tracks):
      if ti not in matched_tracks and tr["time_since_update"] > 0:
        tr["box"] = tr["pred_box"]
        tr["age"] += 1

    # Drop stale tracks
    self._tracks = [tr for tr in self._tracks if tr["time_since_update"] <= self.max_age]

    out: list[TrackedBox] = []
    for tr in self._tracks:
      if tr["hits"] < self.min_hits and tr["time_since_update"] > 0:
        continue
      coasting = tr["time_since_update"] > 0
      out.append(
        TrackedBox(
          track_id=tr["id"],
          box=tuple(tr["box"]),
          score=float(tr["score"]),
          name=tr.get("name"),
          age=int(tr["age"]),
          hits=int(tr["hits"]),
          coasting=coasting,
          time_since_update=int(tr["time_since_update"]),
        )
      )
    return out
