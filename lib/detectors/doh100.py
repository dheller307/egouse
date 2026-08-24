"""100DOH Faster R-CNN wrapper (handobj_100K / handobj_100K+ego)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from lib.detectors.backend import Detection, DetectorBackend
from lib.paths import EGOUSE_ROOT

DOH_ROOT = EGOUSE_ROOT / "third_party" / "Hand_Object_Detector"

# 100DOH contact state indices (demo.py labels)
CONTACT_LABELS = {0: "N", 1: "S", 2: "O", 3: "P", 4: "F"}


class Doh100Backend(DetectorBackend):
  """Wraps ddshan/hand_object_detector demo inference."""

  def __init__(
    self,
    *,
    checkpoint: int = 132028,
    backend_tag: str = "100doh_ego",
    cuda: bool = True,
    thresh_hand: float = 0.5,
    thresh_obj: float = 0.5,
  ) -> None:
    if not DOH_ROOT.is_dir():
      raise FileNotFoundError(f"100DOH repo missing: {DOH_ROOT}")
    if str(DOH_ROOT) not in sys.path:
      sys.path.insert(0, str(DOH_ROOT))

    self._backend_tag = backend_tag
    self._checkpoint = checkpoint
    self._cuda = bool(cuda and torch.cuda.is_available())
    self._thresh_hand = thresh_hand
    self._thresh_obj = thresh_obj
    self._faster_rcnn = None
    self._im_data = None
    self._im_info = None
    self._num_boxes = None
    self._gt_boxes = None
    self._box_info = None
    self._pascal_classes = np.asarray(["__background__", "targetobject", "hand"])
    self._load_model()

  @property
  def backend_name(self) -> str:
    return self._backend_tag

  @property
  def cuda(self) -> bool:
    return self._cuda

  def _load_model(self) -> None:
    import _init_paths  # noqa: F401
    from model.faster_rcnn.resnet import resnet
    from model.utils.config import cfg, cfg_from_file, cfg_from_list
    from model.utils.net_utils import load_net

    cfg_from_file(str(DOH_ROOT / "cfgs" / "res101.yml"))
    cfg_from_list(
      ["ANCHOR_SCALES", "[8, 16, 32, 64]", "ANCHOR_RATIOS", "[0.5, 1, 2]"]
    )
    cfg.USE_GPU_NMS = self._cuda
    if self._cuda:
      cfg.CUDA = True

    model_dir = DOH_ROOT / "models" / "res101_handobj_100K" / "pascal_voc"
    load_name = model_dir / f"faster_rcnn_1_8_{self._checkpoint}.pth"
    if not load_name.is_file():
      raise FileNotFoundError(f"100DOH weights missing: {load_name}")

    faster_rcnn = resnet(self._pascal_classes, 101, pretrained=False, class_agnostic=False)
    faster_rcnn.create_architecture()
    if self._cuda:
      checkpoint = torch.load(str(load_name))
    else:
      checkpoint = torch.load(str(load_name), map_location="cpu")
    faster_rcnn.load_state_dict(checkpoint["model"])
    if "pooling_mode" in checkpoint:
      cfg.POOLING_MODE = checkpoint["pooling_mode"]
    faster_rcnn.eval()
    if self._cuda:
      faster_rcnn.cuda()

    self._faster_rcnn = faster_rcnn
    self._cfg = cfg
    self._im_data = torch.FloatTensor(1)
    self._im_info = torch.FloatTensor(1)
    self._num_boxes = torch.LongTensor(1)
    self._gt_boxes = torch.FloatTensor(1)
    self._box_info = torch.FloatTensor(1)
    if self._cuda:
      self._im_data = self._im_data.cuda()
      self._im_info = self._im_info.cuda()
      self._num_boxes = self._num_boxes.cuda()
      self._gt_boxes = self._gt_boxes.cuda()

  def infer(self, image_bgr) -> list[Detection]:
    from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
    from model.roi_layers import nms
    from model.utils.blob import im_list_to_blob

    cfg = self._cfg
    im = image_bgr.astype(np.float32, copy=True)
    im -= cfg.PIXEL_MEANS
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    processed_ims = []
    im_scale_factors = []
    for target_size in cfg.TEST.SCALES:
      im_scale = float(target_size) / float(im_size_min)
      if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
      resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
      im_scale_factors.append(im_scale)
      processed_ims.append(resized)
    blob = im_list_to_blob(processed_ims)
    im_info_np = np.array([[blob.shape[1], blob.shape[2], im_scale_factors[0]]], dtype=np.float32)
    im_data_pt = torch.from_numpy(blob).permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
      self._im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
      self._im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
      self._gt_boxes.resize_(1, 1, 5).zero_()
      self._num_boxes.resize_(1).zero_()
      self._box_info.resize_(1, 1, 5).zero_()
      rois, cls_prob, bbox_pred, _, _, _, _, _, loss_list = self._faster_rcnn(
        self._im_data, self._im_info, self._gt_boxes, self._num_boxes, self._box_info
      )

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    contact_vector = loss_list[0][0]
    offset_vector = loss_list[1][0].detach()
    lr_vector = loss_list[2][0].detach()
    _, contact_indices = torch.max(contact_vector, 2)
    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()
    lr = (torch.sigmoid(lr_vector) > 0.5).squeeze(0).float()

    if cfg.TEST.BBOX_REG:
      box_deltas = bbox_pred.data
      if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        if self._cuda:
          box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
          box_deltas = box_deltas + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        else:
          box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
          box_deltas = box_deltas + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        box_deltas = box_deltas.view(1, -1, 4 * len(self._pascal_classes))
      pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
      pred_boxes = clip_boxes(pred_boxes, im_info_pt, 1)
    else:
      pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scale_factors[0]
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    out: list[Detection] = []
    for j in range(1, len(self._pascal_classes)):
      cls_name = self._pascal_classes[j]
      thresh = self._thresh_hand if cls_name == "hand" else self._thresh_obj
      inds = torch.nonzero(scores[:, j] > thresh).view(-1)
      if inds.numel() == 0:
        continue
      cls_scores = scores[:, j][inds]
      _, order = torch.sort(cls_scores, 0, True)
      cls_boxes = pred_boxes[inds][:, j * 4 : (j + 1) * 4]
      cls_dets = torch.cat(
        (
          cls_boxes,
          cls_scores.unsqueeze(1),
          contact_indices[inds],
          offset_vector.squeeze(0)[inds],
          lr[inds],
        ),
        1,
      )
      cls_dets = cls_dets[order]
      keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
      cls_dets = cls_dets[keep.view(-1).long()].cpu().numpy()

      label: str = "hand" if cls_name == "hand" else "obj"
      for row in cls_dets:
        x1, y1, x2, y2, score = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
        side = None
        contact = None
        offset = None
        if label == "hand" and row.shape[0] >= 9:
          contact = int(row[5])
          offset = (float(row[6]), float(row[7]))
          side = "right" if float(row[8]) > 0.5 else "left"
        out.append(
          Detection(
            label=label,
            box=(x1, y1, x2, y2),
            score=score,
            side=side,
            contact=contact,
            contact_score=float(contact) if contact is not None else None,
            offset=offset,
          )
        )
    return out
