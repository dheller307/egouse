"""Frozen teacher pipeline: score → cleanup → optional eval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lib.clip import mapping_path_for_sequence, segments_path_for_sequence
from lib.eval import (
  convert_segment_boundaries,
  evaluate_rules,
  label_frames,
  load_segments,
  load_time_mapping,
  validate_segments,
)
from lib.eval_timeline import (
  evaluate_timeline,
  gt_segments_from_timeline_df,
  pred_segments_from_pair_scores,
)
from lib.paths import EGOUSE_ROOT
from lib.scoring.dual_dwell_config import DualDwellPreset, FROZEN_DUAL_DWELL
from lib.scoring.timeline_cleanup import (
  TimelineCleanupConfig,
  cleanup_pair_scores,
  frame_scores_from_clean_pairs,
  segments_dataframe,
)
from lib.viz.timeline import build_timeline_table


FROZEN_CLEANUP_CONFIG = TimelineCleanupConfig(
  max_gap_sec=0.12,
  min_on_sec=0.20,
  backfill_max_sec=0.18,
  remicro_protected_min_on_sec=0.18,
  lone_short_max_duration_sec=1.20,
  lone_neighbor_min_gap_sec=3.0,
  morph_close_sec=0.10,
  morph_open_sec=0.35,
)


@dataclass(frozen=True)
class TeacherPipelineResult:
  sequence_id: str
  output_dir: Path
  preset_name: str
  has_human_gt: bool
  cleanup_stats: dict
  n_pair_rows: int
  n_clean_on: int
  frame_clean: dict | None
  timeline_clean: dict | None


def cleanup_config_from_args(args) -> TimelineCleanupConfig:
  """Build cleanup config from CLI namespace (frozen defaults when unset)."""
  frozen = FROZEN_CLEANUP_CONFIG
  return TimelineCleanupConfig(
    max_gap_sec=getattr(args, "max_gap_sec", frozen.max_gap_sec),
    min_on_sec=getattr(args, "min_on_sec", frozen.min_on_sec),
    backfill_max_sec=getattr(args, "backfill_max_sec", frozen.backfill_max_sec),
    remicro_protected_min_on_sec=getattr(
      args, "remicro_protected_min_on_sec", frozen.remicro_protected_min_on_sec
    ),
    lone_short_max_duration_sec=getattr(
      args, "lone_short_max_duration_sec", frozen.lone_short_max_duration_sec
    ),
    lone_neighbor_min_gap_sec=getattr(
      args, "lone_neighbor_min_gap_sec", frozen.lone_neighbor_min_gap_sec
    ),
    morph_close_sec=getattr(args, "morph_close_sec", frozen.morph_close_sec),
    morph_open_sec=getattr(args, "morph_open_sec", frozen.morph_open_sec),
  )


def ground_truth_paths_for_sequence(sequence_path: Path) -> tuple[Path, Path, bool]:
  """Return (segments_csv, mapping_csv, has_human_gt)."""
  sequence_path = Path(sequence_path)
  segments_path = segments_path_for_sequence(sequence_path, EGOUSE_ROOT)
  mapping_path = mapping_path_for_sequence(sequence_path)
  return segments_path, mapping_path, segments_path.is_file()


def run_frame_eval_strict(
  frame_df: pd.DataFrame,
  gt_path: Path,
  mapping_path: Path,
) -> dict:
  segments_df = load_segments(gt_path)
  validate_segments(segments_df)
  time_mapping_df = load_time_mapping(mapping_path)
  segments_df = convert_segment_boundaries(segments_df, time_mapping_df)
  labeled_df = label_frames(segments_df, frame_df)
  rules = [
    {"name": "v2", "interacting_col": "v2_interacting", "object_col": "v2_object_name"},
  ]
  if "v2_interacting" not in frame_df.columns or "v2_object_name" not in frame_df.columns:
    return {}
  metrics_df = evaluate_rules(frame_df, labeled_df, rules)
  if metrics_df.empty:
    return {}
  row = metrics_df.iloc[0].to_dict()
  return {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.items()}


def run_teacher_pipeline(
  sequence_path: Path,
  object_library_path: Path,
  output_dir: Path,
  *,
  preset: DualDwellPreset = FROZEN_DUAL_DWELL,
  cleanup_cfg: TimelineCleanupConfig | None = None,
  skip_score: bool = False,
  preset_key: str = "frozen",
) -> TeacherPipelineResult:
  """Score (optional), cleanup, write clean artifacts + summary."""
  sequence_path = Path(sequence_path)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  cleanup_cfg = cleanup_cfg or FROZEN_CLEANUP_CONFIG

  gt_path, mapping_path, has_human_gt = ground_truth_paths_for_sequence(sequence_path)

  if not skip_score:
    from lib.experiments.run_gt import run_gt_dual_dwell

    run_gt_dual_dwell(sequence_path, object_library_path, output_dir, preset=preset)
  else:
    pair_path = output_dir / "pair_scores.csv"
    if not pair_path.is_file():
      raise FileNotFoundError(f"Missing {pair_path}; run without skip_score first")

  pair_path = output_dir / "pair_scores.csv"
  if not pair_path.is_file():
    raise FileNotFoundError(f"Missing {pair_path}")

  pair_df = pd.read_csv(pair_path)
  clean, cleanup_stats = cleanup_pair_scores(pair_df, cleanup_cfg)
  frame_clean = frame_scores_from_clean_pairs(clean)
  seg_df = segments_dataframe(pred_segments_from_pair_scores(clean))

  clean.to_csv(output_dir / "pair_scores_clean.csv", index=False)
  frame_clean.to_csv(output_dir / "frame_scores_clean.csv", index=False)
  seg_df.to_csv(output_dir / "segments_pred.csv", index=False)
  (output_dir / "cleanup_stats.json").write_text(
    json.dumps(cleanup_stats, indent=2) + "\n"
  )

  n_clean_on = int(clean["v2_interacting"].astype(bool).sum()) if not clean.empty else 0

  frame_metrics: dict | None = None
  timeline_metrics: dict | None = None

  if has_human_gt:
    frame_clean_path = output_dir / "frame_scores_clean.csv"
    timeline_df = build_timeline_table(frame_clean_path, gt_path, mapping_path)
    pred = pred_segments_from_pair_scores(clean)
    gt = gt_segments_from_timeline_df(timeline_df)
    tl_metrics = evaluate_timeline(gt, pred, iou_threshold=0.5)
    timeline_metrics = {
      "seg_f1": tl_metrics["seg_f1"],
      "seg_prec": tl_metrics["seg_prec"],
      "seg_rec": tl_metrics["seg_rec"],
      "edit_sim": tl_metrics["edit_sim"],
      "n_gt": tl_metrics["overall"]["n_gt"],
      "n_pred": tl_metrics["overall"]["n_pred"],
    }
    (output_dir / "summary_timeline_clean.json").write_text(
      json.dumps(tl_metrics, indent=2) + "\n"
    )

    frame_metrics = run_frame_eval_strict(frame_clean, gt_path, mapping_path)
    (output_dir / "summary_frame_clean.json").write_text(
      json.dumps(frame_metrics, indent=2) + "\n"
    )

  summary: dict = {
    "sequence_id": sequence_path.name,
    "preset": preset.name,
    "preset_key": preset_key,
    "has_human_gt": has_human_gt,
    "cleanup": cleanup_stats,
  }
  if timeline_metrics is not None:
    summary["timeline_clean"] = timeline_metrics
  if frame_metrics is not None:
    summary["frame_clean"] = frame_metrics

  (output_dir / "summary_teacher_timeline.json").write_text(
    json.dumps(summary, indent=2) + "\n"
  )

  return TeacherPipelineResult(
    sequence_id=sequence_path.name,
    output_dir=output_dir,
    preset_name=preset.name,
    has_human_gt=has_human_gt,
    cleanup_stats=cleanup_stats,
    n_pair_rows=len(pair_df),
    n_clean_on=n_clean_on,
    frame_clean=frame_metrics,
    timeline_clean=timeline_metrics,
  )
