"""Shared GRU student training loop (per-hand and merged)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.paths import EGOUSE_ROOT
from lib.student_gru_prove_eval import (
  CHECKPOINT_METRICS,
  prove_full_metrics,
  score_for_checkpoint,
)
from lib.student_gru_model import StudentHandGRU
from lib.student_sequence_data import (
  HandStreamDataset,
  MergedSeqDataset,
  collate_hand_streams,
  collate_merged_seqs,
  compute_class_weights,
  compute_multilabel_pos_weight,
  load_object_vocab,
  load_seq_list,
  save_object_vocab,
  slot_dim_for_hand_cond,
)


def masked_ce_loss(
  logits: torch.Tensor,
  y: torch.Tensor,
  lengths: torch.Tensor,
  criterion: nn.Module,
) -> torch.Tensor:
  b, t, _ = logits.shape
  mask = torch.arange(t, device=logits.device)[None, :] < lengths[:, None]
  flat_logits = logits.reshape(b * t, -1)
  flat_y = y.reshape(b * t)
  flat_mask = mask.reshape(b * t)
  return criterion(flat_logits[flat_mask], flat_y[flat_mask])


def masked_weighted_ce_loss(
  logits: torch.Tensor,
  y: torch.Tensor,
  lengths: torch.Tensor,
  class_weights: torch.Tensor | None,
  sample_w: torch.Tensor | None,
) -> torch.Tensor:
  """Per-timestep CE with optional class and sample weights; mean over valid tokens weighted."""
  b, t, c = logits.shape
  mask = torch.arange(t, device=logits.device)[None, :] < lengths[:, None]
  ce = F.cross_entropy(
    logits.reshape(b * t, c),
    y.reshape(b * t),
    weight=class_weights,
    reduction="none",
    ignore_index=-1,
  ).reshape(b, t)
  if sample_w is None:
    w = mask.float()
  else:
    w = sample_w.float() * mask.float()
  denom = w.sum().clamp_min(1e-8)
  return (ce * w).sum() / denom


def masked_focal_ce_loss(
  logits: torch.Tensor,
  y: torch.Tensor,
  lengths: torch.Tensor,
  class_weights: torch.Tensor | None,
  sample_w: torch.Tensor | None,
  gamma: float,
) -> torch.Tensor:
  """Focal CE (gamma>0) on valid timesteps; optional class/sample weights."""
  b, t, c = logits.shape
  mask = torch.arange(t, device=logits.device)[None, :] < lengths[:, None]
  flat_logits = logits.reshape(b * t, c)
  flat_y = y.reshape(b * t)
  ce = F.cross_entropy(
    flat_logits,
    flat_y,
    weight=class_weights,
    reduction="none",
    ignore_index=-1,
  ).reshape(b, t)
  pt = torch.exp(-ce.clamp_max(50.0))
  focal = (1.0 - pt).pow(gamma) * ce
  if sample_w is None:
    w = mask.float()
  else:
    w = sample_w.float() * mask.float()
  return (focal * w).sum() / w.sum().clamp_min(1e-8)


def masked_spur_suppress_loss(
  logits: torch.Tensor,
  y: torch.Tensor,
  lengths: torch.Tensor,
) -> torch.Tensor:
  """Penalize non-background probability mass on teacher-background timesteps."""
  b, t, c = logits.shape
  mask = torch.arange(t, device=logits.device)[None, :] < lengths[:, None]
  prob = F.softmax(logits, dim=-1)
  non_bg = prob[:, :, 1:].sum(dim=-1) if c > 1 else prob.new_zeros(b, t)
  bg = (y == 0) & mask
  if not bg.any():
    return logits.new_zeros(())
  return non_bg[bg].mean()


def masked_bce_loss(
  logits: torch.Tensor,
  y_multi: torch.Tensor,
  lengths: torch.Tensor,
  pos_weight: torch.Tensor,
) -> torch.Tensor:
  b, t, v = logits.shape
  mask = torch.arange(t, device=logits.device)[None, :, None] < lengths[:, None, None]
  mask = mask.expand(b, t, v)
  pw = pos_weight.view(1, 1, -1)
  weights = torch.where(y_multi > 0.5, pw, torch.ones_like(y_multi))
  loss = F.binary_cross_entropy_with_logits(logits, y_multi, reduction="none")
  return (loss * weights)[mask].mean()


def masked_transition_kl_loss(
  logits: torch.Tensor,
  lengths: torch.Tensor,
  *,
  eps: float = 1e-8,
  multilabel: bool = False,
) -> torch.Tensor:
  _b, t, c = logits.shape
  if t < 2:
    return logits.new_zeros(())
  p = torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=-1)
  p_prev = p[:, :-1, :]
  p_curr = p[:, 1:, :]
  kl_obj = p_prev * (torch.log(p_prev + eps) - torch.log(p_curr + eps))
  kl = kl_obj.mean(dim=-1) if multilabel else kl_obj.sum(dim=-1)
  pair_ok = torch.arange(1, t, device=logits.device)[None, :] < lengths[:, None]
  if not pair_ok.any():
    return logits.new_zeros(())
  return kl[pair_ok].mean()


@torch.no_grad()
def eval_per_hand(
  model: StudentHandGRU,
  loader: DataLoader,
  class_weights: torch.Tensor | None,
  device: torch.device,
) -> tuple[float, float]:
  model.eval()
  total_loss = 0.0
  total_correct = 0
  total_count = 0
  for batch in loader:
    X = batch["X"].to(device)
    y = batch["y"].to(device)
    lengths = batch["lengths"].to(device)
    hand_id = batch.get("hand_id")
    if hand_id is not None:
      hand_id = hand_id.to(device)
    logits = model(X, lengths, hand_id=hand_id)
    loss = masked_weighted_ce_loss(logits, y, lengths, class_weights, None)
    total_loss += float(loss.item()) * lengths.size(0)
    mask = torch.arange(X.size(1), device=device)[None, :] < lengths[:, None]
    pred = logits.argmax(dim=-1)
    total_correct += int(((pred == y) & mask).sum().item())
    total_count += int(mask.sum().item())
  n = max(len(loader.dataset), 1)
  return total_loss / n, total_correct / max(total_count, 1)


@torch.no_grad()
def eval_merged(
  model: StudentHandGRU,
  loader: DataLoader,
  pos_weight: torch.Tensor,
  device: torch.device,
) -> tuple[float, float]:
  model.eval()
  total_loss = 0.0
  total_correct = 0
  total_count = 0
  for batch in loader:
    X = batch["X"].to(device)
    y_multi = batch["y_multi"].to(device)
    lengths = batch["lengths"].to(device)
    logits = model(X, lengths, hand_id=None)
    loss = masked_bce_loss(logits, y_multi, lengths, pos_weight)
    total_loss += float(loss.item()) * lengths.size(0)
    mask = torch.arange(X.size(1), device=device)[None, :, None] < lengths[:, None, None]
    pred = (torch.sigmoid(logits) >= 0.5).float()
    total_correct += int(((pred == y_multi) & mask).sum().item())
    total_count += int(mask.sum().item())
  n = max(len(loader.dataset), 1)
  return total_loss / n, total_correct / max(total_count, 1)


@dataclass
class TrainConfig:
  features_root: Path
  train_list: Path
  val_list: Path
  output_dir: Path
  stream_target: str = "per_hand"
  epochs: int = 40
  batch_size: int = 8
  lr: float = 1e-3
  hidden: int = 128
  d_slot: int = 16
  num_layers: int = 2
  dropout: float = 0.1
  patience: int = 10
  hand_cond: str = "onehot"
  hand_embed_dim: int = 8
  trans_kl_lambda: float = 0.0
  random_state: int = 42
  device: str = "auto"
  sample_weights: dict[tuple[str, str], np.ndarray] | None = None
  init_state_dict: dict[str, torch.Tensor] | None = None
  checkpoint_metric: str = "val_ce"  # val_ce | prove_timeline | prove_prec | prove_f05 | prove_prec_penalized
  teacher_root: Path | None = None
  prove_seg_weight: float = 3.0
  prove_edit_weight: float = 1.0
  prove_iou_threshold: float = 0.5
  prove_spur_penalty: float = 0.15
  class_weight_cap: float | None = None
  use_class_weights: bool = True
  bg_class_weight_mult: float = 1.0
  spur_suppress_lambda: float = 0.0
  focal_gamma: float = 0.0
  decode_margin: float = 0.0
  decode_min_prob: float = 0.0
  augment_train: bool = False
  feature_jitter: float = 0.0
  feature_channel_dropout: float = 0.0


def resolve_device(device_str: str) -> torch.device:
  if device_str == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.device(device_str)


def train_student_gru(cfg: TrainConfig) -> dict[str, Any]:
  if cfg.stream_target == "merged" and cfg.hand_cond != "none":
    cfg.hand_cond = "none"

  torch.manual_seed(cfg.random_state)
  np.random.seed(cfg.random_state)
  device = resolve_device(cfg.device)

  object_vocab, object_to_id = load_object_vocab()
  vocab_size = len(object_vocab)
  multilabel = cfg.stream_target == "merged"
  num_classes = vocab_size if multilabel else vocab_size + 1
  label_mode = "multilabel" if multilabel else "exclusive"

  train_seqs = load_seq_list(cfg.train_list)
  val_seqs = load_seq_list(cfg.val_list)

  if multilabel:
    train_ds = MergedSeqDataset(train_seqs, cfg.features_root, object_vocab, object_to_id)
    val_ds = MergedSeqDataset(val_seqs, cfg.features_root, object_vocab, object_to_id)
    collate_fn = collate_merged_seqs
  else:
    train_ds = HandStreamDataset(
      train_seqs,
      cfg.features_root,
      object_vocab,
      object_to_id,
      hand_cond=cfg.hand_cond,
      sample_weights=cfg.sample_weights,
      augment_train=cfg.augment_train,
      feature_jitter=cfg.feature_jitter,
      feature_channel_dropout=cfg.feature_channel_dropout,
      rng=np.random.default_rng(cfg.random_state),
    )
    val_ds = HandStreamDataset(
      val_seqs, cfg.features_root, object_vocab, object_to_id, hand_cond=cfg.hand_cond
    )
    collate_fn = collate_hand_streams

  train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
  )
  val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
  )

  slot_dim = slot_dim_for_hand_cond(cfg.hand_cond)
  class_weights_tensor: torch.Tensor | None = None
  if multilabel:
    print("Computing multilabel pos_weight on train sequences...", flush=True)
    pos_weight = compute_multilabel_pos_weight(train_ds).to(device)
    print("pos_weight done.", flush=True)
  else:
    if cfg.use_class_weights:
      print("Computing class weights on train streams...", flush=True)
      class_weights_tensor = compute_class_weights(
        train_ds, num_classes, cap=cfg.class_weight_cap
      )
      if cfg.bg_class_weight_mult != 1.0:
        class_weights_tensor = class_weights_tensor.clone()
        class_weights_tensor[0] *= cfg.bg_class_weight_mult
      class_weights_tensor = class_weights_tensor.to(device)
    else:
      print("Skipping inverse-freq class weights (uniform CE).", flush=True)
      class_weights_tensor = None
    pos_weight = None

  model = StudentHandGRU(
    vocab_size=vocab_size,
    slot_dim=slot_dim,
    d_slot=cfg.d_slot,
    hidden=cfg.hidden,
    num_layers=cfg.num_layers,
    dropout=cfg.dropout,
    num_classes=num_classes,
    hand_cond=cfg.hand_cond,
    hand_embed_dim=cfg.hand_embed_dim,
    label_mode=label_mode,
  ).to(device)
  if cfg.init_state_dict is not None:
    model.load_state_dict(cfg.init_state_dict)

  optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  save_object_vocab(cfg.output_dir / "object_vocab.json", object_vocab)

  best_val = float("inf")
  best_prove_score = float("-inf")
  best_epoch = -1
  stale = 0
  best_path = cfg.output_dir / "best.pt"
  last_val_acc = float("nan")
  last_prove_seg = float("nan")
  last_prove_edit = float("nan")
  best_prove_seg = float("nan")
  best_prove_edit = float("nan")
  best_prove_metrics: dict[str, float] = {}
  use_prove_ckpt = cfg.checkpoint_metric in CHECKPOINT_METRICS and cfg.checkpoint_metric != "val_ce"
  if use_prove_ckpt and cfg.teacher_root is None:
    cfg.teacher_root = EGOUSE_ROOT / "outputs" / "teacher"

  for epoch in range(1, cfg.epochs + 1):
    model.train()
    train_loss_sum = 0.0
    train_trans_sum = 0.0
    for batch in train_loader:
      X = batch["X"].to(device)
      lengths = batch["lengths"].to(device)
      optim.zero_grad(set_to_none=True)
      hand_id = batch.get("hand_id")
      if hand_id is not None:
        hand_id = hand_id.to(device)
      logits = model(X, lengths, hand_id=hand_id)

      if multilabel:
        y_multi = batch["y_multi"].to(device)
        loss_main = masked_bce_loss(logits, y_multi, lengths, pos_weight)
      else:
        y = batch["y"].to(device)
        sample_w = batch.get("sample_w")
        if sample_w is not None:
          sample_w = sample_w.to(device)
        if cfg.focal_gamma > 0:
          loss_main = masked_focal_ce_loss(
            logits, y, lengths, class_weights_tensor, sample_w, cfg.focal_gamma
          )
        else:
          loss_main = masked_weighted_ce_loss(
            logits, y, lengths, class_weights_tensor, sample_w
          )

      loss = loss_main
      trans_term = logits.new_zeros(())
      spur_term = logits.new_zeros(())
      if not multilabel and cfg.spur_suppress_lambda > 0:
        spur_term = masked_spur_suppress_loss(logits, y, lengths)
        loss = loss + cfg.spur_suppress_lambda * spur_term
      if cfg.trans_kl_lambda > 0:
        trans_term = masked_transition_kl_loss(logits, lengths, multilabel=multilabel)
        loss = loss_main + cfg.trans_kl_lambda * trans_term
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optim.step()
      train_loss_sum += float(loss_main.item()) * lengths.size(0)
      if cfg.trans_kl_lambda > 0:
        train_trans_sum += float(trans_term.item()) * lengths.size(0)

    train_loss = train_loss_sum / max(len(train_ds), 1)
    val_loss, val_acc = (
      eval_merged(model, val_loader, pos_weight, device)
      if multilabel
      else eval_per_hand(model, val_loader, class_weights_tensor, device)
    )
    last_val_acc = val_acc
    prove_seg = prove_edit = prove_score = float("nan")
    prove_metrics_dict: dict[str, float] = {}
    if use_prove_ckpt:
      pm = prove_full_metrics(
        model,
        object_vocab,
        object_to_id,
        device,
        val_seqs,
        cfg.features_root,
        cfg.teacher_root,
        iou_threshold=cfg.prove_iou_threshold,
        seg_weight=cfg.prove_seg_weight,
        edit_weight=cfg.prove_edit_weight,
        spur_penalty=cfg.prove_spur_penalty,
        decode_margin=cfg.decode_margin,
        decode_min_prob=cfg.decode_min_prob,
      )
      prove_metrics_dict = pm.as_dict()
      prove_seg = pm.mean_seg_f1
      prove_edit = pm.mean_edit_sim
      prove_score = score_for_checkpoint(
        cfg.checkpoint_metric,
        prove_metrics_dict,
        spur_penalty=cfg.prove_spur_penalty,
      )
      last_prove_seg = prove_seg
      last_prove_edit = prove_edit
    loss_name = "bce" if multilabel else "ce"
    if cfg.trans_kl_lambda > 0:
      train_trans = train_trans_sum / max(len(train_ds), 1)
      line = (
        f"epoch {epoch:3d}  train_{loss_name}={train_loss:.4f}  train_kl={train_trans:.4f}  "
        f"val_{loss_name}={val_loss:.4f}  val_acc={val_acc:.4f}"
      )
    else:
      line = (
        f"epoch {epoch:3d}  train_{loss_name}={train_loss:.4f}  "
        f"val_{loss_name}={val_loss:.4f}  val_acc={val_acc:.4f}"
      )
    if use_prove_ckpt and np.isfinite(prove_score):
      line += (
        f"  prove_seg={prove_seg:.4f}  prove_edit={prove_edit:.4f}  "
        f"prove_prec={prove_metrics_dict.get('mean_seg_prec', float('nan')):.4f}  "
        f"spur={prove_metrics_dict.get('spur_ratio', float('nan')):.4f}  "
        f"prove_score={prove_score:.4f}"
      )
    print(line, flush=True)

    improved = False
    if use_prove_ckpt and np.isfinite(prove_score):
      if prove_score > best_prove_score + 1e-5:
        best_prove_score = prove_score
        best_prove_seg = prove_seg
        best_prove_edit = prove_edit
        best_prove_metrics = prove_metrics_dict
        best_epoch = epoch
        stale = 0
        improved = True
    elif np.isfinite(val_loss) and val_loss < best_val:
      best_val = val_loss
      best_epoch = epoch
      stale = 0
      improved = True

    if improved:
      torch.save(
        {
          "model_state": model.state_dict(),
          "object_vocab": object_vocab,
          "object_to_id": object_to_id,
          "hparams": {
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "slot_dim": slot_dim,
            "hand_cond": cfg.hand_cond,
            "hand_embed_dim": cfg.hand_embed_dim,
            "trans_kl_lambda": cfg.trans_kl_lambda,
            "stream_target": cfg.stream_target,
            "label_mode": label_mode,
            "d_slot": cfg.d_slot,
            "hidden": cfg.hidden,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
          },
        },
        best_path,
      )
    else:
      stale += 1
      if stale >= cfg.patience:
        print(f"early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
        break

  hparams_dict = {
    "features_root": str(cfg.features_root),
    "train_list": str(cfg.train_list),
    "val_list": str(cfg.val_list),
    "output_dir": str(cfg.output_dir),
    "stream_target": cfg.stream_target,
    "epochs": cfg.epochs,
    "batch_size": cfg.batch_size,
    "lr": cfg.lr,
    "hidden": cfg.hidden,
    "d_slot": cfg.d_slot,
    "num_layers": cfg.num_layers,
    "dropout": cfg.dropout,
    "patience": cfg.patience,
    "hand_cond": cfg.hand_cond,
    "hand_embed_dim": cfg.hand_embed_dim,
    "trans_kl_lambda": cfg.trans_kl_lambda,
    "random_state": cfg.random_state,
    "device": str(device),
    "used_sample_weights": cfg.sample_weights is not None,
    "warm_start": cfg.init_state_dict is not None,
    "checkpoint_metric": cfg.checkpoint_metric,
    "prove_seg_weight": cfg.prove_seg_weight,
    "prove_edit_weight": cfg.prove_edit_weight,
    "prove_spur_penalty": cfg.prove_spur_penalty,
    "class_weight_cap": cfg.class_weight_cap,
    "use_class_weights": cfg.use_class_weights,
    "bg_class_weight_mult": cfg.bg_class_weight_mult,
    "spur_suppress_lambda": cfg.spur_suppress_lambda,
    "focal_gamma": cfg.focal_gamma,
    "decode_margin": cfg.decode_margin,
    "decode_min_prob": cfg.decode_min_prob,
    "augment_train": cfg.augment_train,
    "feature_jitter": cfg.feature_jitter,
    "feature_channel_dropout": cfg.feature_channel_dropout,
  }
  meta = {
    "best_epoch": best_epoch,
    "best_val_loss": best_val if np.isfinite(best_val) and not use_prove_ckpt else None,
    "best_prove_score": best_prove_score if use_prove_ckpt and np.isfinite(best_prove_score) else None,
    "prove_seg_at_best": best_prove_seg if use_prove_ckpt else None,
    "prove_edit_at_best": best_prove_edit if use_prove_ckpt else None,
    "prove_metrics_at_best": best_prove_metrics if use_prove_ckpt else None,
    "checkpoint_metric": cfg.checkpoint_metric,
    "val_acc_at_best": last_val_acc,
    "n_train_units": len(train_ds),
    "n_val_units": len(val_ds),
    "num_classes": num_classes,
    "label_mode": label_mode,
    "stream_target": cfg.stream_target,
    "device": str(device),
    "hparams": hparams_dict,
    "model_path": str(best_path),
  }
  (cfg.output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str))
  print(json.dumps(meta, indent=2))
  print(f"wrote {best_path}")
  return meta
