"""Shared helpers for dual-dwell experiment runners."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lib.clip import mapping_path_for_sequence, resolve_clip_window, segments_path_for_sequence
from lib.eval import (
  convert_segment_boundaries,
  evaluate_rules,
  label_frames,
  load_segments,
  load_time_mapping,
  validate_segments,
)
from lib.experiments.registry import get_motion_frames
from lib.paths import EGOUSE_ROOT
from lib.scoring.dual_dwell_config import DualDwellPreset, FROZEN_DUAL_DWELL
from lib.scoring.interaction_v2 import V2DwellTracker
from lib.scoring.pipeline import ScoringConfig
from lib.viz.timeline import add_clip_t_sec

ARIA_RGB_IMAGE_HEIGHT = 1408
V2_RULE = {"name": "v2", "interacting_col": "v2_interacting", "object_col": "v2_object_name"}


def fire_rate(df: pd.DataFrame, col: str = "v2_interacting") -> float:
  if df.empty or col not in df.columns:
    return 0.0
  return 100.0 * df[col].astype(bool).sum() / len(df)


def cm_ok_rate(df: pd.DataFrame) -> float:
  if df.empty or "v2_co_motion_ok" not in df.columns:
    return 0.0
  return 100.0 * df["v2_co_motion_ok"].astype(bool).sum() / len(df)


def to_scored_df(rows: list[dict]) -> pd.DataFrame:
  if not rows:
    return pd.DataFrame()
  df = pd.DataFrame(rows)
  for col in df.columns:
    if "interacting" in col or col.endswith("_ok"):
      df[col] = df[col].astype(int)
  return add_clip_t_sec(df)


def new_v2_tracker(config: ScoringConfig) -> V2DwellTracker:
  return V2DwellTracker(
    iou_dwell_ms=config.v2_iou_dwell_ms,
    cm_dwell_ms=config.v2_co_motion_dwell_ms,
    iou_gap_tolerance_ms=config.v2_iou_gap_tolerance_ms,
    cm_gap_tolerance_ms=config.v2_co_motion_gap_tolerance_ms,
  )


def init_scoring_context(sequence_path: Path, object_library_path: Path):
  """Load HOT3D 2D boxes + clip window (shared by gt and mps paths)."""
  from lib.hot3d_io import init_provider

  sequence_path = Path(sequence_path)
  segments_path = segments_path_for_sequence(sequence_path, EGOUSE_ROOT)
  mapping_path = mapping_path_for_sequence(sequence_path)
  has_human_gt = segments_path.is_file()

  hot3d, _, timestamps, hand_prov, obj_prov, obj_lib = init_provider(
    sequence_path, object_library_path
  )
  clip = resolve_clip_window(
    timestamps,
    start_frame=None,
    num_frames=None,
    stride=1,
    segments_path=segments_path,
    mapping_path=mapping_path,
  )
  clip_mode = "GT bounds" if has_human_gt else "full VRS"
  print(f"Clip ({clip_mode}): {clip.num_frames} frames, {clip.duration_sec:.1f}s")

  time_slice = timestamps[
    clip.start_frame : clip.start_frame + clip.num_frames * clip.stride : clip.stride
  ]
  return {
    "hot3d": hot3d,
    "hand_prov": hand_prov,
    "obj_prov": obj_prov,
    "obj_lib": obj_lib,
    "time_slice": time_slice,
    "segments_path": segments_path,
    "mapping_path": mapping_path,
    "has_human_gt": has_human_gt,
  }


def evaluate_frame(df: pd.DataFrame, frame_key: str, segments_path: Path, mapping_path: Path) -> pd.DataFrame:
  segments_df = load_segments(segments_path)
  validate_segments(segments_df)
  mapping_df = load_time_mapping(mapping_path)
  segments_df = convert_segment_boundaries(segments_df, mapping_df)
  labeled = label_frames(segments_df, df)
  metrics = evaluate_rules(df, labeled, [V2_RULE])
  metrics["motion_frame"] = get_motion_frames()[frame_key].eval_label
  return metrics


def pairwise_vs_gt(df: pd.DataFrame, df_gt: pd.DataFrame, suffix: str) -> dict[str, int]:
  merged = df.merge(
    df_gt,
    on=["timestamp_ns", "hand_side"],
    suffixes=(f"_{suffix}", "_gt"),
    how="outer",
  )
  col_a = f"v2_interacting_{suffix}"
  col_b = "v2_interacting_gt"
  if col_a not in merged.columns or col_b not in merged.columns:
    return {}
  a = merged[col_a].astype(bool)
  b = merged[col_b].astype(bool)
  return {
    "both_on": int((a & b).sum()),
    f"{suffix}_only_on": int((a & ~b).sum()),
    "gt_only_on": int((~a & b).sum()),
    "both_off": int((~a & ~b).sum()),
  }


def write_frame_outputs(
  output_dir: Path,
  frame_key: str,
  df: pd.DataFrame,
  metrics: pd.DataFrame,
  preset: DualDwellPreset,
  *,
  extra_config: dict | None = None,
  pair_df: pd.DataFrame | None = None,
) -> dict:
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  df.to_csv(output_dir / "frame_scores.csv", index=False)
  if pair_df is not None and not pair_df.empty:
    pair_df.to_csv(output_dir / "pair_scores.csv", index=False)
    print(f"Wrote {len(pair_df)} pair rows -> {output_dir / 'pair_scores.csv'}")

  spec = get_motion_frames()[frame_key]
  frame_summary = {
    "motion_frame": spec.eval_label,
    "label": spec.label,
    "fire_rate_pct": fire_rate(df),
    "cm_ok_rate_pct": cm_ok_rate(df),
    "num_rows": len(df),
  }
  if pair_df is not None and not pair_df.empty:
    frame_summary["num_pair_rows"] = len(pair_df)
    frame_summary["pair_fire_rate_pct"] = fire_rate(pair_df)
  if not metrics.empty:
    frame_summary["metrics"] = metrics.iloc[0].to_dict()
  (output_dir / "summary.json").write_text(json.dumps(frame_summary, indent=2) + "\n")

  payload = {
    "preset": preset.name,
    "description": preset.description,
    "motion_frame": frame_key,
    **preset.config_dict(),
    **{k: getattr(preset.to_scoring_config(), k) for k in (
      "visibility_threshold",
      "iou_threshold",
      "dist_threshold",
      "spatial_threshold",
      "engagement_threshold",
      "tot_min_ms",
    )},
    **(extra_config or {}),
  }
  (output_dir / "experiment_config.json").write_text(json.dumps(payload, indent=2) + "\n")
  print(f"Wrote {len(df)} rows -> {output_dir / 'frame_scores.csv'}")
  return frame_summary


def print_frame_report(frame_key: str, df: pd.DataFrame, metrics: pd.DataFrame) -> None:
  spec = get_motion_frames()[frame_key]
  print(f"\n=== {spec.label} ===")
  print(f"  fire {fire_rate(df):5.1f}%   cm_ok {cm_ok_rate(df):5.1f}%")
  if not metrics.empty:
    row = metrics.iloc[0]
    print(f"  prec {row['prec']:.3f}  rec {row['rec']:.3f}  s_f1 {row['s_f1']:.3f}")
