"""Teacher dual-dwell v2 knobs (gt_world recall-tuned on P0003)."""

from dataclasses import dataclass
from typing import Optional

from lib.scoring.pipeline import ScoringConfig


@dataclass(frozen=True)
class DualDwellPreset:
  """Named experiment preset for v2 dual-dwell scoring."""

  name: str
  description: str
  spatial_threshold: float = 0.3
  co_motion_threshold: float = 0.4
  co_motion_v_min_px: float = 30.0
  co_motion_v_min_m_s: float = 0.01
  v2_iou_dwell_ms: float = 99.0
  v2_co_motion_dwell_ms: float = 33.0
  v2_iou_gap_tolerance_ms: float = 130.0
  v2_co_motion_gap_tolerance_ms: float = 300.0
  # Stationary OR-branch (gt_world): AABB surface proximity + low 3D speed
  enable_stationary: bool = True
  stationary_v_max_m_s: float = 0.015
  stationary_aabb_d_max_m: float = 0.12
  # None → same as stationary_v_max_m_s (symmetric frozen default)
  stationary_hand_v_max_m_s: Optional[float] = None
  # Opt-in mid-band (slow rotate / near-hold). Default OFF — experimental only.
  enable_slow_manip: bool = False
  slow_manip_v_max_m_s: float = 0.06
  # Opt-in: co-motion proximity via AABB instead of 2D IoU. Default OFF — A/B only.
  enable_aabb_co_motion: bool = False
  # None → use stationary_aabb_d_max_m for AABB-cm proximity
  co_motion_aabb_d_max_m: Optional[float] = None

  def effective_co_motion_aabb_d_max_m(self) -> float:
    if self.co_motion_aabb_d_max_m is not None:
      return self.co_motion_aabb_d_max_m
    return self.stationary_aabb_d_max_m

  def effective_stationary_hand_v_max_m_s(self) -> float:
    if self.stationary_hand_v_max_m_s is not None:
      return self.stationary_hand_v_max_m_s
    return self.stationary_v_max_m_s

  def to_scoring_config(self) -> ScoringConfig:
    return ScoringConfig(
      visibility_threshold=0.3,
      iou_threshold=0.05,
      dist_threshold=0.45,
      spatial_threshold=self.spatial_threshold,
      k=3.0,
      w_spatial=0.9,
      w_approach=0.1,
      engagement_threshold=0.35,
      tot_min_ms=165.0,
      max_history_len=15,
      smooth_frames=5,
      min_var_frames=5,
      gap_reset_ms=200.0,
      gap_tolerance_ms=99.0,
      velocity_threshold=250.0,
      norm_variance_threshold=0.002,
      manipulation_scale=1.0,
      idle_scale=0.8,
      co_motion_threshold=self.co_motion_threshold,
      co_motion_v_min=self.co_motion_v_min_px,
      v2_iou_dwell_ms=self.v2_iou_dwell_ms,
      v2_co_motion_dwell_ms=self.v2_co_motion_dwell_ms,
      v2_iou_gap_tolerance_ms=self.v2_iou_gap_tolerance_ms,
      v2_co_motion_gap_tolerance_ms=self.v2_co_motion_gap_tolerance_ms,
    )

  def config_dict(self) -> dict:
    """JSON-serializable preset fields."""
    return {
      "name": self.name,
      "description": self.description,
      "spatial_threshold": self.spatial_threshold,
      "co_motion_threshold": self.co_motion_threshold,
      "co_motion_v_min_px": self.co_motion_v_min_px,
      "co_motion_v_min_m_s": self.co_motion_v_min_m_s,
      "v2_iou_dwell_ms": self.v2_iou_dwell_ms,
      "v2_co_motion_dwell_ms": self.v2_co_motion_dwell_ms,
      "v2_iou_gap_tolerance_ms": self.v2_iou_gap_tolerance_ms,
      "v2_co_motion_gap_tolerance_ms": self.v2_co_motion_gap_tolerance_ms,
      "enable_stationary": self.enable_stationary,
      "stationary_v_max_m_s": self.stationary_v_max_m_s,
      "stationary_aabb_d_max_m": self.stationary_aabb_d_max_m,
      "stationary_hand_v_max_m_s": self.stationary_hand_v_max_m_s,
      "enable_slow_manip": self.enable_slow_manip,
      "slow_manip_v_max_m_s": self.slow_manip_v_max_m_s,
      "enable_aabb_co_motion": self.enable_aabb_co_motion,
      "co_motion_aabb_d_max_m": self.co_motion_aabb_d_max_m,
      "co_motion_dims": "xyz",
    }


FROZEN_DUAL_DWELL = DualDwellPreset(
  name="teacher_aabb_stationary_v22",
  description=(
    "gt_world teacher: IoU+XYZ co-motion OR class-free AABB-surface stationary "
    "(d_max=0.12 m wrist-to-surface, v_max=0.015 m/s). Per-object interacting. "
    "Slow-manip and AABB co-motion proximity available via experimental flags (default OFF)."
  ),
)

# Experimental A/B only — not the default teacher.
EXPERIMENTAL_SLOW_MANIP = DualDwellPreset(
  name="teacher_aabb_slow_manip_exp",
  description=(
    "Same as teacher_aabb_stationary_v22 plus opt-in near-hold/slow-manip "
    "(aabb≤0.12 m, both speeds < 0.06 m/s). For A/B only."
  ),
  enable_slow_manip=True,
  slow_manip_v_max_m_s=0.06,
)

# Experimental A/B only — co-motion proximity via AABB instead of IoU.
EXPERIMENTAL_AABB_CO_MOTION = DualDwellPreset(
  name="teacher_aabb_co_motion_exp",
  description=(
    "AABB co-motion proximity (d=0.16 m) with tighter stationary (d=0.10 m) "
    "and asymmetric hand v_max=0.04 m/s. For A/B only."
  ),
  enable_aabb_co_motion=True,
  stationary_aabb_d_max_m=0.10,
  stationary_hand_v_max_m_s=0.04,
  co_motion_aabb_d_max_m=0.16,
)

# Desk FP crackdown: tighter stationary proximity + looser hand jitter (2a+2b).
EXPERIMENTAL_STATIONARY_DESK = DualDwellPreset(
  name="teacher_stationary_desk_exp",
  description=(
    "Tight stationary AABB d=0.10 m; object v_max=0.015 m/s, hand v_max=0.04 m/s. "
    "IoU co-motion unchanged. For A/B only."
  ),
  stationary_aabb_d_max_m=0.10,
  stationary_hand_v_max_m_s=0.04,
)

# Alias kept for older imports / MPS contrast runs.
TEACHER_DUAL_DWELL = FROZEN_DUAL_DWELL
