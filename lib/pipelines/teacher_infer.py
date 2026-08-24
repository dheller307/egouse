"""Teacher pipeline wrapper (gt_world → y*_clean)."""

from __future__ import annotations

from pathlib import Path

from lib.experiments.teacher_pipeline import (
  FROZEN_CLEANUP_CONFIG,
  TeacherPipelineResult,
  cleanup_config_from_args,
  run_teacher_pipeline,
)
from lib.paths import EGOUSE_ROOT, HOT3D_DATA_DIR
from lib.scoring.dual_dwell_config import (
  EXPERIMENTAL_AABB_CO_MOTION,
  EXPERIMENTAL_SLOW_MANIP,
  EXPERIMENTAL_STATIONARY_DESK,
  FROZEN_DUAL_DWELL,
  DualDwellPreset,
)

PRESETS: dict[str, DualDwellPreset] = {
  "frozen": FROZEN_DUAL_DWELL,
  "aabb_cm_exp": EXPERIMENTAL_AABB_CO_MOTION,
  "stationary_desk_exp": EXPERIMENTAL_STATIONARY_DESK,
  "slow_manip_exp": EXPERIMENTAL_SLOW_MANIP,
}

DEFAULT_OUTPUT_ROOT = EGOUSE_ROOT / "outputs" / "teacher"


def resolve_sequence_path(sequence: str) -> Path:
  p = Path(sequence)
  if p.is_absolute() and p.is_dir():
    return p
  candidate = HOT3D_DATA_DIR / sequence
  if candidate.is_dir():
    return candidate
  raise FileNotFoundError(f"Sequence not found: {sequence}")


def run_teacher_for_sequence(
  sequence: str,
  *,
  output_dir: Path | None = None,
  preset_key: str = "frozen",
  skip_score: bool = False,
  object_library_path: Path | None = None,
) -> TeacherPipelineResult:
  """Run frozen teacher pipeline for one HOT3D sequence."""
  seq_path = resolve_sequence_path(sequence)
  out_dir = output_dir or (DEFAULT_OUTPUT_ROOT / seq_path.name)
  preset = PRESETS.get(preset_key, FROZEN_DUAL_DWELL)
  assets = object_library_path or (HOT3D_DATA_DIR / "assets")
  return run_teacher_pipeline(
    seq_path,
    assets,
    out_dir,
    preset=preset,
    cleanup_cfg=FROZEN_CLEANUP_CONFIG,
    skip_score=skip_score,
    preset_key=preset_key,
  )
