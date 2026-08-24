#!/usr/bin/env python3
"""Train causal GRU student on per-hand or merged multi-label streams."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.paths import EGOUSE_ROOT
from lib.student_gru_train_core import TrainConfig, train_student_gru
from lib.student_sequence_data import STREAM_TARGET_CHOICES

DEFAULT_FEATURES_ROOT = EGOUSE_ROOT / "outputs" / "student" / "features_raw"
DEFAULT_OUT = EGOUSE_ROOT / "outputs" / "student" / "gru_v1_onehot"


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Train causal GRU HOI student")
  p.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
  p.add_argument(
    "--train-list",
    type=Path,
    default=EGOUSE_ROOT / "config" / "seq_lists" / "train.txt",
  )
  p.add_argument(
    "--val-list",
    type=Path,
    default=EGOUSE_ROOT / "config" / "seq_lists" / "prove.txt",
  )
  p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
  p.add_argument(
    "--stream-target",
    choices=STREAM_TARGET_CHOICES,
    default="per_hand",
    help="per_hand: exclusive CE per hand; merged: one stream/seq, multi-label objects",
  )
  p.add_argument("--epochs", type=int, default=40)
  p.add_argument("--batch-size", type=int, default=8)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--hidden", type=int, default=128)
  p.add_argument("--d-slot", type=int, default=16)
  p.add_argument("--num-layers", type=int, default=2)
  p.add_argument("--dropout", type=float, default=0.1)
  p.add_argument("--patience", type=int, default=10)
  p.add_argument(
    "--hand-cond",
    choices=("onehot", "embed", "none"),
    default="onehot",
    help="Hand identity (per_hand only); merged target uses pooled features (none)",
  )
  p.add_argument("--hand-embed-dim", type=int, default=8)
  p.add_argument(
    "--trans-kl-lambda",
    type=float,
    default=0.0,
    help="Masked KL between consecutive step distributions (softmax or sigmoid)",
  )
  p.add_argument("--random-state", type=int, default=42)
  p.add_argument("--device", type=str, default="auto")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  if args.stream_target == "merged" and args.hand_cond != "none":
    print("stream-target merged: forcing hand-cond none (pooled L/R features)", flush=True)
  cfg = TrainConfig(
    features_root=args.features_root,
    train_list=args.train_list,
    val_list=args.val_list,
    output_dir=args.output_dir,
    stream_target=args.stream_target,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    hidden=args.hidden,
    d_slot=args.d_slot,
    num_layers=args.num_layers,
    dropout=args.dropout,
    patience=args.patience,
    hand_cond=args.hand_cond,
    hand_embed_dim=args.hand_embed_dim,
    trans_kl_lambda=args.trans_kl_lambda,
    random_state=args.random_state,
    device=args.device,
  )
  train_student_gru(cfg)


if __name__ == "__main__":
  main()
