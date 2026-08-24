"""GT-world dual-dwell runner (fast — no MPS load)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lib.experiments._common import (
  ARIA_RGB_IMAGE_HEIGHT,
  evaluate_frame,
  fire_rate,
  init_scoring_context,
  new_v2_tracker,
  print_frame_report,
  to_scored_df,
  write_frame_outputs,
)
from lib.motion.world_motion import WorldMotionHistory
from lib.motion.object_aabb import ObjectAabbLibrary
from lib.pipelines.gt_world import score_frame_v2_gt_world
from lib.scoring.dual_dwell_config import DualDwellPreset, FROZEN_DUAL_DWELL


def run_gt_dual_dwell(
  sequence_path: Path,
  object_library_path: Path,
  output_dir: Path,
  preset: DualDwellPreset = FROZEN_DUAL_DWELL,
) -> pd.DataFrame:
  config = preset.to_scoring_config()
  ctx = init_scoring_context(sequence_path, object_library_path)

  hand_pose_prov = ctx["hot3d"].umetrack_hand_data_provider
  object_pose_prov = ctx["hot3d"].object_pose_data_provider
  if hand_pose_prov is None:
    raise RuntimeError("UmeTrack hand pose provider unavailable (required for gt_world)")

  hand_world = WorldMotionHistory(
    smooth_frames=config.smooth_frames, gap_reset_ms=config.gap_reset_ms
  )
  obj_world = WorldMotionHistory(
    smooth_frames=config.smooth_frames, gap_reset_ms=config.gap_reset_ms
  )
  tracker = new_v2_tracker(config)
  object_aabb_library = ObjectAabbLibrary(ctx["obj_lib"].asset_folder_name)
  rows: list[dict] = []
  pair_rows: list[dict] = []
  prev_ts = None

  for frame_index, ts in enumerate(ctx["time_slice"]):
    dt_ns = 33_300_000 if prev_ts is None else ts - prev_ts
    prev_ts = ts
    frame_rows, frame_pairs = score_frame_v2_gt_world(
      ts,
      frame_index,
      ctx["hand_prov"],
      ctx["obj_prov"],
      hand_pose_prov,
      object_pose_prov,
      ctx["obj_lib"],
      ARIA_RGB_IMAGE_HEIGHT,
      config,
      hand_world,
      obj_world,
      tracker,
      dt_ns,
      co_motion_v_min_m_s=preset.co_motion_v_min_m_s,
      enable_stationary=preset.enable_stationary,
      stationary_v_max_m_s=preset.stationary_v_max_m_s,
      stationary_aabb_d_max_m=preset.stationary_aabb_d_max_m,
      stationary_hand_v_max_m_s=preset.stationary_hand_v_max_m_s,
      enable_slow_manip=preset.enable_slow_manip,
      slow_manip_v_max_m_s=preset.slow_manip_v_max_m_s,
      enable_aabb_co_motion=preset.enable_aabb_co_motion,
      co_motion_aabb_d_max_m=preset.co_motion_aabb_d_max_m,
      object_aabb_library=object_aabb_library,
    )
    rows.extend(frame_rows)
    pair_rows.extend(frame_pairs)

  df = to_scored_df(rows)
  pair_df = to_scored_df(pair_rows)
  if ctx["has_human_gt"]:
    metrics = evaluate_frame(df, "gt", ctx["segments_path"], ctx["mapping_path"])
    print_frame_report("gt", df, metrics)
  else:
    metrics = pd.DataFrame()
    print("\n=== GT world (no human segments — skip frame eval) ===")
    print(f"  fire {fire_rate(df):5.1f}%")
  write_frame_outputs(output_dir, "gt", df, metrics, preset, pair_df=pair_df)
  return df
