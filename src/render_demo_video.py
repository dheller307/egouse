#!/usr/bin/env python3
"""Render demo MP4: RGB + detector boxes + student/teacher swimlane (see NEXT_AGENT_DEMO_VIDEO.md)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.paths import (
  EGOUSE_ROOT,
  HOT3D_DATA_DIR,
  resolve_student_checkpoint,
  setup_paths,
)

setup_paths()

import cv2
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch

from lib.hot3d_io import RGB_STREAM_ID, init_provider
from lib.student_gru_infer import load_gru_checkpoint, pair_scores_for_sequence
from lib.viz.timeline import merge_hands_pair_df, plot_teacher_student_hoi_figure, render_hoi_timeline_image

DEFAULT_FEATURES_ROOT = EGOUSE_ROOT / "outputs" / "student" / "features_raw"
DEFAULT_TEACHER_ROOT = EGOUSE_ROOT / "outputs" / "teacher"
DEFAULT_FINAL_DIR = EGOUSE_ROOT / "outputs" / "plots" / "final"

# Poster-scale timeline (match outputs/plots/final/*_50s.png crops).
TIMELINE_FIGSIZE = (8.5, 4.8)
TIMELINE_DPI = 200
OBJECT_FONTSIZE = 18
TIME_LABELSIZE = 16
PANEL_TITLE_FONTSIZE = 15
TICK_SEC = 5.0
BAR_HEIGHT = 0.82
DASH_LINEWIDTH = 2.4
# Timeline height as fraction of video height (slightly shorter than RGB panel).
TIMELINE_HEIGHT_FRAC = 0.95


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Demo MP4: RGB + detector boxes + student/teacher timeline")
  p.add_argument("--sequence", type=str, default="P0010_0ecbf39f")
  p.add_argument("--model", type=Path, default=None)
  p.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
  p.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
  p.add_argument("--t-min", type=float, default=50.0)
  p.add_argument("--t-max", type=float, default=90.0)
  p.add_argument("--fps", type=float, default=15.0)
  p.add_argument("--output", type=Path, default=None)
  p.add_argument("--device", type=str, default="auto")
  p.add_argument("--score-thresh", type=float, default=0.25, help="Min detector score to draw box")
  p.add_argument("--static-only", action="store_true", help="Write static PNG only (phase 1 sanity)")
  p.add_argument("--max-frames", type=int, default=None, help="Cap frames (debug)")
  return p.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open() as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def _load_preds_by_ts(path: Path) -> dict[int, dict]:
  return {int(rec["timestamp_ns"]): rec for rec in _load_jsonl(path)}


def _t0_ns_from_features(features_path: Path) -> int:
  df = pd.read_csv(features_path, usecols=["timestamp_ns", "t_sec"])
  at_zero = df.loc[df["t_sec"] == 0.0, "timestamp_ns"]
  if not at_zero.empty:
    return int(at_zero.iloc[0])
  return int(df["timestamp_ns"].min())


def _select_frame_records(
  preds_rows: list[dict],
  *,
  t0_ns: int,
  t_min: float,
  t_max: float,
  fps: float,
) -> list[dict]:
  in_window = [
    rec
    for rec in preds_rows
    if t_min <= (int(rec["timestamp_ns"]) - t0_ns) / 1e9 <= t_max
  ]
  if not in_window:
    raise ValueError(f"No preds in t_sec window [{t_min}, {t_max}]")
  n_target = max(1, int(round((t_max - t_min) * fps)))
  if len(in_window) <= n_target:
    return in_window
  idx = np.linspace(0, len(in_window) - 1, n_target).astype(int)
  return [in_window[i] for i in idx]


def _load_student_teacher_merged(
  seq: str,
  model_path: Path,
  features_root: Path,
  teacher_root: Path,
  device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  feat_path = features_root / seq / "features.csv"
  teacher_path = teacher_root / seq / "pair_scores_clean.csv"
  if not feat_path.is_file():
    raise FileNotFoundError(feat_path)
  if not teacher_path.is_file():
    raise FileNotFoundError(teacher_path)
  if not model_path.is_file():
    raise FileNotFoundError(model_path)

  model, vocab, obj_to_id, hparams = load_gru_checkpoint(model_path, device)
  student = pair_scores_for_sequence(seq, features_root, model, vocab, obj_to_id, device)
  teacher = pd.read_csv(teacher_path)[
    ["timestamp_ns", "t_sec", "hand_side", "object_name", "v2_interacting"]
  ].copy()

  if hparams.get("label_mode") == "multilabel":
    student = merge_hands_pair_df(student)
    teacher = merge_hands_pair_df(teacher)
  else:
    student = merge_hands_pair_df(student)
    teacher = merge_hands_pair_df(teacher)
  return student, teacher


def _draw_ltrb_box(
  img: np.ndarray,
  box: list[float],
  color_bgr: tuple[int, int, int],
  label: str,
  *,
  score: float | None = None,
) -> None:
  x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
  cv2.rectangle(img, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), color_bgr, 2)
  text = f"{label} {score:.2f}" if score is not None else label
  cv2.putText(
    img,
    text,
    (x0, max(18, y0 - 4)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    color_bgr,
    2,
    cv2.LINE_AA,
  )


def _draw_detector_overlay(
  img_bgr: np.ndarray,
  pred_rec: dict | None,
  *,
  score_thresh: float,
) -> np.ndarray:
  out = img_bgr.copy()
  if pred_rec is None:
    return out
  seen_sides: set[str] = set()
  for hand in pred_rec.get("hands") or []:
    score = float(hand.get("score", 0.0))
    if score < score_thresh:
      continue
    side = str(hand.get("side") or "hand")
    if side in seen_sides:
      side = f"{side}*"
    seen_sides.add(str(hand.get("side") or "hand"))
    color = (0, 220, 0) if str(hand.get("side")) == "left" else (0, 140, 255)
    _draw_ltrb_box(out, hand["box"], color, side, score=score)
  for obj in pred_rec.get("objects") or []:
    score = float(obj.get("score", 0.0))
    if score < score_thresh:
      continue
    _draw_ltrb_box(out, obj["box"], (40, 40, 240), str(obj.get("name", "obj")), score=score)
  return out


def _get_rgb_frame(device_data_provider, timestamp_ns: int) -> np.ndarray:
  image_rgb = device_data_provider.get_image(int(timestamp_ns), RGB_STREAM_ID)
  image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
  return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)


def _frame_size_for_panels(vid_h: int, vid_w: int, tl_h: int, tl_w: int) -> tuple[int, int]:
  """Video left; timeline right at natural aspect, height ≈ TIMELINE_HEIGHT_FRAC × video."""
  tl_disp_h = max(1, int(round(vid_h * TIMELINE_HEIGHT_FRAC)))
  tl_disp_w = max(1, int(round(tl_disp_h * tl_w / tl_h)))
  return (vid_w + tl_disp_w, vid_h)


def _composite_frame(rgb_bgr: np.ndarray, timeline_rgb: np.ndarray) -> np.ndarray:
  """Side-by-side: full-size RGB left; timeline right, aspect-preserving, vertically centered."""
  vid_h, vid_w = rgb_bgr.shape[:2]
  tl_h, tl_w = timeline_rgb.shape[:2]
  tl_disp_h = max(1, int(round(vid_h * TIMELINE_HEIGHT_FRAC)))
  tl_disp_w = max(1, int(round(tl_disp_h * tl_w / tl_h)))

  timeline_bgr = cv2.cvtColor(timeline_rgb, cv2.COLOR_RGB2BGR)
  tl_resized = cv2.resize(timeline_bgr, (tl_disp_w, tl_disp_h), interpolation=cv2.INTER_AREA)

  composite = np.zeros((vid_h, vid_w + tl_disp_w, 3), dtype=np.uint8)
  composite[:, :vid_w] = rgb_bgr
  y0 = (vid_h - tl_disp_h) // 2
  composite[y0 : y0 + tl_disp_h, vid_w : vid_w + tl_disp_w] = tl_resized
  return composite


def _write_static_png(
  student: pd.DataFrame,
  teacher: pd.DataFrame,
  *,
  seq: str,
  t_min: float,
  t_max: float,
  output: Path,
) -> None:
  output.parent.mkdir(parents=True, exist_ok=True)
  plot_teacher_student_hoi_figure(
    student,
    teacher,
    tick_sec=TICK_SEC,
    output_path=output,
    title=f"{seq} — GRU student (fill) vs teacher y_clean (dashed)",
    hands=("merged",),
    fill_label="student",
    dashed_label="teacher",
    t_min=t_min,
    t_max=t_max,
    object_fontsize=OBJECT_FONTSIZE,
    time_labelsize=TIME_LABELSIZE,
    title_fontsize=OBJECT_FONTSIZE,
    panel_title_fontsize=PANEL_TITLE_FONTSIZE,
    figsize=TIMELINE_FIGSIZE,
    bar_height=BAR_HEIGHT,
    dash_linewidth=DASH_LINEWIDTH,
    dpi=TIMELINE_DPI,
  )


def _open_video_writer(path: Path, frame_size: tuple[int, int], fps: float) -> cv2.VideoWriter:
  path.parent.mkdir(parents=True, exist_ok=True)
  writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
  if not writer.isOpened():
    raise RuntimeError(f"Failed to open video writer for {path}")
  return writer


def main() -> None:
  args = parse_args()
  seq = args.sequence
  model_path = args.model or resolve_student_checkpoint()
  t_min, t_max = float(args.t_min), float(args.t_max)

  device = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "auto"
    else torch.device(args.device)
  )

  feat_dir = args.features_root / seq
  preds_path = feat_dir / "preds.jsonl"
  features_path = feat_dir / "features.csv"
  if not preds_path.is_file():
    raise FileNotFoundError(preds_path)

  student, teacher = _load_student_teacher_merged(
    seq, model_path, args.features_root, args.teacher_root, device
  )

  default_out = DEFAULT_FINAL_DIR / f"demo_{seq}_{int(t_min)}_{int(t_max)}s.mp4"
  static_out = DEFAULT_FINAL_DIR / f"demo_{seq}_{int(t_min)}_{int(t_max)}s_static.png"
  output = args.output or (static_out if args.static_only else default_out)

  if args.static_only:
    _write_static_png(student, teacher, seq=seq, t_min=t_min, t_max=t_max, output=output)
    print(f"Wrote static {output}")
    print(f"checkpoint={model_path}")
    return

  t0_ns = _t0_ns_from_features(features_path)
  preds_rows = _load_jsonl(preds_path)
  frame_records = _select_frame_records(
    preds_rows, t0_ns=t0_ns, t_min=t_min, t_max=t_max, fps=args.fps
  )
  if args.max_frames is not None:
    frame_records = frame_records[: args.max_frames]

  preds_by_ts = _load_preds_by_ts(preds_path)
  seq_path = HOT3D_DATA_DIR / seq
  object_library_path = HOT3D_DATA_DIR / "assets"
  _, device_data_provider, _, _, _, _ = init_provider(seq_path, object_library_path)

  probe_ts = int(frame_records[0]["timestamp_ns"])
  probe_rgb = _get_rgb_frame(device_data_provider, probe_ts)
  vid_h, vid_w = probe_rgb.shape[:2]
  probe_tl = render_hoi_timeline_image(
    student,
    teacher,
    t_min=t_min,
    t_max=t_max,
    t_cursor=(probe_ts - t0_ns) / 1e9,
    tick_sec=TICK_SEC,
    object_fontsize=OBJECT_FONTSIZE,
    time_labelsize=TIME_LABELSIZE,
    panel_title_fontsize=PANEL_TITLE_FONTSIZE,
    figsize=TIMELINE_FIGSIZE,
    bar_height=BAR_HEIGHT,
    dash_linewidth=DASH_LINEWIDTH,
    dpi=TIMELINE_DPI,
    show_title=False,
  )
  tl_h, tl_w = probe_tl.shape[:2]
  frame_size = _frame_size_for_panels(vid_h, vid_w, tl_h, tl_w)
  print(f"frame_size={frame_size[0]}x{frame_size[1]} video={vid_w}x{vid_h} timeline_aspect={tl_w/tl_h:.2f}")

  writer = _open_video_writer(output, frame_size, args.fps)
  t_first = (int(frame_records[0]["timestamp_ns"]) - t0_ns) / 1e9
  t_last = (int(frame_records[-1]["timestamp_ns"]) - t0_ns) / 1e9

  for i, rec in enumerate(frame_records):
    ts = int(rec["timestamp_ns"])
    t_sec = (ts - t0_ns) / 1e9
    rgb = _get_rgb_frame(device_data_provider, ts)
    rgb = _draw_detector_overlay(rgb, preds_by_ts.get(ts, rec), score_thresh=args.score_thresh)
    timeline_rgb = render_hoi_timeline_image(
      student,
      teacher,
      t_min=t_min,
      t_max=t_max,
      t_cursor=t_sec,
      tick_sec=TICK_SEC,
      object_fontsize=OBJECT_FONTSIZE,
      time_labelsize=TIME_LABELSIZE,
      panel_title_fontsize=PANEL_TITLE_FONTSIZE,
      figsize=TIMELINE_FIGSIZE,
      bar_height=BAR_HEIGHT,
      dash_linewidth=DASH_LINEWIDTH,
      dpi=TIMELINE_DPI,
      show_title=False,
    )
    composite = _composite_frame(rgb, timeline_rgb)
    writer.write(composite)
    if (i + 1) % 50 == 0 or i + 1 == len(frame_records):
      print(f"frame {i + 1}/{len(frame_records)} t={t_sec:.2f}s")

  writer.release()
  print(f"Wrote {output}")
  print(
    f"frames={len(frame_records)} fps={args.fps} "
    f"t_sec=[{t_first:.2f}, {t_last:.2f}] checkpoint={model_path}"
  )


if __name__ == "__main__":
  main()
