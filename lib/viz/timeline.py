import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from lib.paths import EGOUSE_ROOT, setup_paths

setup_paths()

from lib.eval import (
  load_segments,
  load_time_mapping,
  validate_segments,
  convert_segment_boundaries,
  label_frames,
  timecode_to_device,
)
from lib.eval_timeline import on_intervals
from lib.viz.timeline_signal import process_confidence

RULE_PLOT_SPECS: dict[str, dict[str, str]] = {
  "iou": {"score_col": "iou_score", "threshold_key": "iou_threshold", "ylabel": "IoU"},
  "dist": {"score_col": "dist_score", "threshold_key": "dist_threshold", "ylabel": "dist conf"},
  "spatial": {"score_col": "spatial_score", "threshold_key": "spatial_threshold", "ylabel": "spatial conf"},
  "engagement": {
    "score_col": "engagement_score",
    "threshold_key": "engagement_threshold",
    "ylabel": "engagement",
  },
  "tot": {"score_col": "tot_ms", "threshold_key": "tot_min_ms", "ylabel": "tot / tot_min"},
  "v2": {
    "score_col": "v2_dwell_ms",
    "threshold_key": "v2_iou_dwell_ms",
    "ylabel": "v2 dwell / min",
  },
}

TOT_YMAX = 2.5
DEFAULT_TICK_SEC = 10.0


def timeline_t_range(timeline_df: pd.DataFrame) -> tuple[float, float]:
  t = timeline_df["t_sec"]
  return float(t.min()), float(t.max())


def configure_timeline_xaxis(
  ax,
  t_min: float,
  t_max: float,
  *,
  tick_sec: float = DEFAULT_TICK_SEC,
  show_labels: bool = True,
  labelsize: float = 8,
) -> None:
  """Shared time axis: major ticks every tick_sec, minor at half interval."""
  pad = max(0.5, (t_max - t_min) * 0.01) if (t_max - t_min) > 1.0 else 0.25
  # Exact window when caller already set a display crop (no extra pad beyond range).
  if t_max - t_min <= 50.0:
    ax.set_xlim(t_min, t_max)
  else:
    ax.set_xlim(t_min - pad, t_max + pad)
  ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_sec))
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:.0f}"))
  if tick_sec >= 2:
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(tick_sec / 2))
  ax.tick_params(axis="x", labelbottom=show_labels, labelsize=labelsize)
  ax.grid(True, axis="x", alpha=0.35, which="major")
  ax.grid(True, axis="x", alpha=0.15, which="minor")


def apply_timeline_time_axes(
  axes,
  t_min: float,
  t_max: float,
  *,
  tick_sec: float = DEFAULT_TICK_SEC,
  time_labelsize: float = 8,
  xlabel_fontsize: float = 10,
) -> None:
  """Put readable time ticks on every stacked row (not only the bottom)."""
  for ax in axes:
    configure_timeline_xaxis(
      ax, t_min, t_max, tick_sec=tick_sec, show_labels=True, labelsize=time_labelsize
    )
  axes[-1].set_xlabel("t (s from clip start)", fontsize=xlabel_fontsize)


def load_scoring_config(config_path: Path | None = None) -> dict:
  path = config_path or (EGOUSE_ROOT / "outputs" / "interactions" / "scoring_config.json")
  return json.loads(Path(path).read_text())


def build_timeline_table(
  frame_scores_path: Path,
  ground_truths_path: Path,
  mapping_path: Path,
) -> pd.DataFrame:
  frame_scores_df = pd.read_csv(frame_scores_path)
  segments_df = load_segments(ground_truths_path)
  validate_segments(segments_df)

  mapping_path = Path(mapping_path)
  time_mapping_df = load_time_mapping(mapping_path)

  segments_df = convert_segment_boundaries(segments_df, time_mapping_df)
  labeled_df = label_frames(segments_df, frame_scores_df)
  timeline_df = frame_scores_df.merge(
    labeled_df,
    on=["timestamp_ns", "hand_side"],
    how="left",
    validate="one_to_one",
  )

  return add_timeline_time_columns(timeline_df, time_mapping_df)


def add_clip_t_sec(
  frame_scores_df: pd.DataFrame,
  t0_ns: int | None = None,
) -> pd.DataFrame:
  """Seconds from first row's timestamp_ns; matches timeline plot x-axis."""
  if t0_ns is None:
    t0_ns = int(frame_scores_df["timestamp_ns"].min())
  out = frame_scores_df.copy()
  t_sec = (out["timestamp_ns"] - t0_ns) / 1e9
  if "t_sec" in out.columns:
    out["t_sec"] = t_sec
    return out
  insert_at = 1 if "frame_index" in out.columns else 1
  out.insert(insert_at, "t_sec", t_sec)
  return out


def add_timeline_time_columns(
  timeline_df: pd.DataFrame,
  mapping_df: pd.DataFrame,
  t0_ns: int | None = None,
) -> pd.DataFrame:
  if t0_ns is None:
    t0_ns = int(timeline_df["timestamp_ns"].min())
  copy_timeline_df = timeline_df.copy()
  copy_timeline_df["t_sec"] = (copy_timeline_df["timestamp_ns"] - t0_ns) / 1e9
  copy_timeline_df["devicetime_ns"] = copy_timeline_df["timestamp_ns"].apply(
    lambda tc: timecode_to_device(int(tc), mapping_df)
  )
  return copy_timeline_df


def hand_timeline_df(timeline_df: pd.DataFrame, hand_side: str) -> pd.DataFrame:
  return (
    timeline_df[timeline_df["hand_side"] == hand_side]
    .sort_values("t_sec")
    .reset_index(drop=True)
  )


def rule_plot_series(
  hand_df: pd.DataFrame,
  score_col: str,
  threshold: float,
) -> tuple[np.ndarray, np.ndarray, float, str]:
  """
  Return (t_sec, y_raw, plot_threshold, ylabel) for plotting.

  tot_ms is normalized to tot_ms / tot_min_ms so threshold plots at 1.0.
  """
  t = hand_df["t_sec"].to_numpy()
  y_raw = hand_df[score_col].to_numpy(dtype=float)
  ylabel = score_col
  for spec in RULE_PLOT_SPECS.values():
    if spec["score_col"] == score_col:
      ylabel = spec["ylabel"]
      break

  if score_col == "tot_ms":
    if threshold <= 0:
      raise ValueError("tot_min_ms must be > 0 for tot normalization")
    y_raw = y_raw / threshold
    plot_threshold = 1.0
  elif score_col == "v2_dwell_ms":
    if threshold <= 0:
      raise ValueError("v2_dwell_ms must be > 0 for v2 normalization")
    y_raw = y_raw / threshold
    plot_threshold = 1.0
  else:
    plot_threshold = threshold

  return t, y_raw, plot_threshold, ylabel


def plot_gt_panel(
  ax,
  hand_df: pd.DataFrame,
  *,
  hand_label: str = "",
) -> None:
  """GT-only row in stacked layout."""
  if hand_df.empty:
    ax.set_title(f"{hand_label} — no rows")
    return

  t = hand_df["t_sec"].to_numpy()
  y = hand_df["gt_interacting"].astype(bool).astype(float).to_numpy()
  ax.fill_between(t, 0, y, step="post", color="tab:green", alpha=0.4, label="GT interacting")
  ax.step(t, y, where="post", color="tab:green", linewidth=1.0)
  ax.set_ylim(-0.05, 1.05)
  ax.set_yticks([0, 1])
  ax.set_yticklabels(["off", "on"])
  ax.set_ylabel("GT")
  ax.set_title(f"{hand_label} — ground truth")
  ax.legend(loc="upper right", fontsize=8)


def plot_rule_panel(
  ax,
  hand_df: pd.DataFrame,
  score_col: str,
  *,
  threshold: float,
  ylabel: str,
  smooth_window: int = 5,
  gap_sec: float = 0.0,
  dwell_sec: float = 0.0,
  hand_label: str = "",
) -> None:
  """Rule row: shaded processed-on regions + raw/smoothed curves (no GT overlay)."""
  if hand_df.empty:
    ax.set_title(f"{hand_label} — no rows")
    return

  if score_col not in hand_df.columns:
    raise KeyError(f"Column {score_col!r} not in timeline table; re-run teacher scoring")

  if score_col in ("tot_ms", "v2_dwell_ms"):
    smooth_window = 1

  t, y_raw, plot_threshold, ylabel = rule_plot_series(hand_df, score_col, threshold)
  y_smooth, y_binary = process_confidence(
    t,
    y_raw,
    smooth_window=smooth_window,
    threshold=plot_threshold,
    gap_sec=gap_sec,
    dwell_sec=dwell_sec,
  )

  is_tot = score_col == "tot_ms"
  is_v2_dwell = score_col == "v2_dwell_ms"
  if is_tot or is_v2_dwell:
    ymax = TOT_YMAX
    y_plot_raw = np.minimum(y_raw, ymax)
    y_plot_smooth = np.minimum(y_smooth, ymax)
    shade_top = ymax
  else:
    ymax = 1.05
    y_plot_raw = y_raw
    y_plot_smooth = y_smooth
    shade_top = 1.0

  if y_binary is not None:
    ax.fill_between(
      t,
      0,
      y_binary * shade_top,
      step="post",
      color="tab:blue",
      alpha=0.2,
      label="predicted on",
    )

  ax.plot(t, y_plot_raw, color="tab:gray", alpha=0.45, linewidth=0.8, label="raw")
  ax.plot(t, y_plot_smooth, color="tab:orange", linewidth=1.5, label="smoothed")
  ax.axhline(plot_threshold, color="gray", linestyle="--", linewidth=1, label="threshold")

  ax.set_ylim(-0.05, ymax)

  ax.set_ylabel(ylabel)
  ax.set_title(f"{hand_label} — {ylabel}")
  ax.legend(loc="upper right", fontsize=8)


def plot_gt_timeline_figure(
  timeline_df: pd.DataFrame,
  *,
  tick_sec: float = DEFAULT_TICK_SEC,
  output_path: Path | None = None,
  hands: tuple[str, ...] = ("left", "right"),
) -> plt.Figure:
  """GT-only timeline (one subplot per hand)."""
  t_min, t_max = timeline_t_range(timeline_df)
  fig, axes = plt.subplots(len(hands), 1, figsize=(14, 2.5 * len(hands)), sharex=True)
  if len(hands) == 1:
    axes = [axes]

  for ax, hand in zip(axes, hands):
    plot_gt_panel(ax, hand_timeline_df(timeline_df, hand), hand_label=f"{hand} hand")

  apply_timeline_time_axes(axes, t_min, t_max, tick_sec=tick_sec)
  fig.tight_layout()

  if output_path is not None:
    _save_figure(fig, output_path)

  return fig


def plot_rule_timeline_figure(
  timeline_df: pd.DataFrame,
  rule: str,
  *,
  scoring_config: dict | None = None,
  smooth_window: int = 5,
  gap_sec: float = 0.13,
  dwell_sec: float = 0.099,
  tick_sec: float = DEFAULT_TICK_SEC,
  layout: str = "stacked",
  output_path: Path | None = None,
  hands: tuple[str, ...] = ("left", "right"),
) -> plt.Figure:
  """
  Timeline for one rule.

  layout='stacked' (default): per hand, GT row then rule row (shared x).
  layout='overlay': legacy single row per hand with GT under curves.
  """
  if rule not in RULE_PLOT_SPECS:
    raise ValueError(f"Unknown rule {rule!r}; choose from {list(RULE_PLOT_SPECS)}")

  spec = RULE_PLOT_SPECS[rule]
  score_col = spec["score_col"]
  config = scoring_config or load_scoring_config()
  threshold = float(config[spec["threshold_key"]])
  ylabel = spec["ylabel"]
  t_min, t_max = timeline_t_range(timeline_df)

  if layout == "stacked":
    row_count = len(hands) * 2
    fig, axes = plt.subplots(row_count, 1, figsize=(14, 2.2 * row_count), sharex=True)
    if row_count == 1:
      axes = [axes]

    row = 0
    for hand in hands:
      hand_df = hand_timeline_df(timeline_df, hand)
      hand_label = f"{hand} hand"
      plot_gt_panel(axes[row], hand_df, hand_label=hand_label)
      row += 1
      plot_rule_panel(
        axes[row],
        hand_df,
        score_col,
        threshold=threshold,
        ylabel=ylabel,
        smooth_window=smooth_window,
        gap_sec=gap_sec,
        dwell_sec=dwell_sec,
        hand_label=hand_label,
      )
      row += 1

    apply_timeline_time_axes(axes, t_min, t_max, tick_sec=tick_sec)
    fig.suptitle(
      f"{rule}: smooth={smooth_window}, gap={gap_sec}s, dwell={dwell_sec}s",
      y=1.005,
      fontsize=11,
    )
  elif layout == "overlay":
    fig, axes = plt.subplots(len(hands), 1, figsize=(14, 2.5 * len(hands)), sharex=True)
    if len(hands) == 1:
      axes = [axes]

    for ax, hand in zip(axes, hands):
      hand_df = hand_timeline_df(timeline_df, hand)
      plot_rule_panel(
        ax,
        hand_df,
        score_col,
        threshold=threshold,
        ylabel=ylabel,
        smooth_window=smooth_window,
        gap_sec=gap_sec,
        dwell_sec=dwell_sec,
        hand_label=f"{hand} hand",
      )
      t = hand_df["t_sec"].to_numpy()
      y_gt = hand_df["gt_interacting"].astype(bool).astype(float).to_numpy()
      ax.fill_between(t, 0, y_gt, step="post", color="tab:green", alpha=0.15, label="GT")
      ax.legend(loc="upper right", fontsize=8)

    apply_timeline_time_axes(axes, t_min, t_max, tick_sec=tick_sec)
    fig.suptitle(f"timeline: {rule} (overlay)", y=1.01)
  else:
    raise ValueError(f"Unknown layout {layout!r}; use 'stacked' or 'overlay'")

  fig.tight_layout()

  if output_path is not None:
    _save_figure(fig, output_path)

  return fig


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=150, bbox_inches="tight")


# Distinct colors for HOI object lanes (demo-friendly, not rainbow-noise).
_HOI_OBJECT_COLORS: dict[str, str] = {
  "keyboard": "#4C78A8",
  "mouse": "#F58518",
  "mug_white": "#E45756",
  "coffee_pot": "#72B7B2",
  "puzzle_toy": "#54A24B",
  "cellphone": "#B279A2",
}


def _object_color(name: str, fallback_idx: int = 0) -> str:
  if name in _HOI_OBJECT_COLORS:
    return _HOI_OBJECT_COLORS[name]
  palette = list(_HOI_OBJECT_COLORS.values())
  return palette[fallback_idx % len(palette)]


def plot_per_object_hoi_panel(
  ax,
  hand_pair_df: pd.DataFrame,
  gt_hand_df: pd.DataFrame | None,
  *,
  hand_label: str = "",
  object_order: list[str] | None = None,
) -> None:
  """
  Swimlane timeline: one lane per object with predicted ON bars; GT hatch overlay.
  """
  if hand_pair_df.empty:
    ax.set_title(f"{hand_label} — no pair rows")
    return

  names = object_order or sorted(hand_pair_df["object_name"].unique())
  # Prefer objects that ever interact or appear in GT
  active = []
  for name in names:
    sub = hand_pair_df[hand_pair_df["object_name"] == name]
    gt_hit = False
    if gt_hand_df is not None and not gt_hand_df.empty:
      gt_hit = bool((gt_hand_df["gt_object_name"] == name).any())
    if gt_hit or bool(sub["v2_interacting"].astype(bool).any()):
      active.append(name)
  if not active:
    active = names[:6]

  y_positions = {name: i for i, name in enumerate(reversed(active))}
  for name in active:
    y = y_positions[name]
    color = _object_color(name, y)
    sub = (
      hand_pair_df[hand_pair_df["object_name"] == name]
      .sort_values("t_sec")
      .reset_index(drop=True)
    )
    t = sub["t_sec"].to_numpy(dtype=float)
    on = sub["v2_interacting"].astype(bool).to_numpy()
    for t0, t1 in on_intervals(t, on):
      ax.barh(
        y,
        t1 - t0,
        left=t0,
        height=0.7,
        color=color,
        alpha=0.85,
        edgecolor="none",
      )

    if gt_hand_df is not None and not gt_hand_df.empty:
      gt = gt_hand_df.sort_values("t_sec").reset_index(drop=True)
      gt_on = (gt["gt_object_name"] == name).to_numpy()
      gt_t = gt["t_sec"].to_numpy(dtype=float)
      for t0, t1 in on_intervals(gt_t, gt_on):
        ax.barh(
          y,
          t1 - t0,
          left=t0,
          height=0.7,
          facecolor="none",
          edgecolor="black",
          linewidth=1.2,
          linestyle="--",
          alpha=0.9,
        )

  ax.set_yticks(list(y_positions.values()))
  ax.set_yticklabels(list(y_positions.keys()), fontsize=9)
  ax.set_ylim(-0.6, len(active) - 0.4)
  ax.set_ylabel(hand_label)
  ax.set_title(f"{hand_label} — per-object HOI (fill=pred, dashed=GT)")


def _object_order_from_pair_dfs(*dfs: pd.DataFrame, object_order: list[str] | None = None) -> list[str]:
  preferred = [
    "keyboard",
    "mouse",
    "mug_white",
    "coffee_pot",
    "puzzle_toy",
    "cellphone",
    "whiteboard_marker",
    "whiteboard_eraser",
  ]
  if object_order is not None:
    return list(object_order)
  present: set[str] = set()
  for df in dfs:
    if df is not None and not df.empty:
      present.update(str(n) for n in df["object_name"].unique())
  order = [n for n in preferred if n in present]
  order.extend(n for n in sorted(present) if n not in order)
  return order


def gt_pair_df_from_timeline(
  template_pair_df: pd.DataFrame,
  timeline_df: pd.DataFrame,
) -> pd.DataFrame:
  """
  Human GT as pair_scores rows on the teacher grid.

  One row per (timestamp_ns, hand_side, object_name) in template_pair_df;
  v2_interacting is ON when labeled gt_object_name matches object_name.
  """
  need = {"timestamp_ns", "hand_side", "object_name"}
  missing = need - set(template_pair_df.columns)
  if missing:
    raise KeyError(f"template_pair_df missing columns: {sorted(missing)}")
  tl_need = {"timestamp_ns", "hand_side", "gt_object_name", "gt_interacting"}
  tl_missing = tl_need - set(timeline_df.columns)
  if tl_missing:
    raise KeyError(f"timeline_df missing columns: {sorted(tl_missing)}")

  base = template_pair_df[list(need)].copy()
  if "t_sec" in template_pair_df.columns:
    base["t_sec"] = template_pair_df["t_sec"]
  gt_cols = timeline_df[list(tl_need)].drop_duplicates(
    subset=["timestamp_ns", "hand_side"], keep="first"
  )
  merged = base.merge(gt_cols, on=["timestamp_ns", "hand_side"], how="left", validate="many_to_one")
  merged["v2_interacting"] = (
    merged["gt_interacting"].fillna(False).astype(bool)
    & (merged["gt_object_name"].astype(str) == merged["object_name"].astype(str))
  ).astype(int)
  if "t_sec" not in merged.columns:
    t0 = int(merged["timestamp_ns"].min())
    merged["t_sec"] = (merged["timestamp_ns"] - t0) / 1e9
  return merged[
    ["timestamp_ns", "t_sec", "hand_side", "object_name", "v2_interacting"]
  ].reset_index(drop=True)


def merge_hands_pair_df(pair_df: pd.DataFrame) -> pd.DataFrame:
  """
  Union left/right into one timeline: per (timestamp_ns, object_name), ON if any hand is ON.

  Multiple objects may be ON at the same timestamp (different swimlanes).
  """
  if pair_df.empty:
    return pair_df.copy()
  need = {"timestamp_ns", "object_name", "v2_interacting"}
  missing = need - set(pair_df.columns)
  if missing:
    raise KeyError(f"pair_df missing columns: {sorted(missing)}")
  work = pair_df.copy()
  if "t_sec" not in work.columns:
    t0 = int(work["timestamp_ns"].min())
    work["t_sec"] = (work["timestamp_ns"] - t0) / 1e9
  agg: dict[str, str] = {"v2_interacting": "max", "t_sec": "first"}
  if "student_prob" in work.columns:
    agg["student_prob"] = "max"
  merged = (
    work.groupby(["timestamp_ns", "object_name"], as_index=False)
    .agg(agg)
    .sort_values(["timestamp_ns", "object_name"])
    .reset_index(drop=True)
  )
  merged["hand_side"] = "merged"
  return merged


def _pair_on_in_window(df: pd.DataFrame, name: str, t0: float, t1: float) -> bool:
  if df.empty:
    return False
  sub = df[df["object_name"] == name]
  if sub.empty:
    return False
  m = (sub["t_sec"].astype(float) >= t0) & (sub["t_sec"].astype(float) <= t1)
  return bool(sub.loc[m, "v2_interacting"].astype(bool).any())


def plot_teacher_student_hoi_panel(
  ax,
  student_pair_df: pd.DataFrame,
  teacher_pair_df: pd.DataFrame,
  *,
  hand_label: str = "",
  object_order: list[str] | None = None,
  fill_label: str = "student",
  dashed_label: str = "teacher",
  object_fontsize: float = 9,
  title_fontsize: float = 11,
  window: tuple[float, float] | None = None,
  bar_height: float = 0.7,
  dash_linewidth: float = 1.2,
) -> None:
  """Swimlanes: student ON (fill) vs teacher y_clean (dashed outline)."""
  if student_pair_df.empty and teacher_pair_df.empty:
    ax.set_title(f"{hand_label} — no pair rows")
    return

  order = _object_order_from_pair_dfs(student_pair_df, teacher_pair_df, object_order=object_order)
  w0, w1 = window if window is not None else (float("-inf"), float("inf"))
  active = []
  for name in order:
    if window is not None:
      s_on = _pair_on_in_window(student_pair_df, name, w0, w1)
      t_on = _pair_on_in_window(teacher_pair_df, name, w0, w1)
    else:
      s_on = False
      t_on = False
      if not student_pair_df.empty:
        sub = student_pair_df[student_pair_df["object_name"] == name]
        s_on = bool(sub["v2_interacting"].astype(bool).any())
      if not teacher_pair_df.empty:
        sub = teacher_pair_df[teacher_pair_df["object_name"] == name]
        t_on = bool(sub["v2_interacting"].astype(bool).any())
    if s_on or t_on:
      active.append(name)
  if not active:
    active = order[:6]

  y_positions = {name: i for i, name in enumerate(reversed(active))}
  for name in active:
    y = y_positions[name]
    color = _object_color(name, y)
    if not student_pair_df.empty:
      sub = (
        student_pair_df[student_pair_df["object_name"] == name]
        .sort_values("t_sec")
        .reset_index(drop=True)
      )
      t = sub["t_sec"].to_numpy(dtype=float)
      on = sub["v2_interacting"].astype(bool).to_numpy()
      for a0, a1 in on_intervals(t, on):
        ax.barh(
          y,
          a1 - a0,
          left=a0,
          height=bar_height,
          color=color,
          alpha=0.75,
          edgecolor="none",
        )

    if not teacher_pair_df.empty:
      sub = (
        teacher_pair_df[teacher_pair_df["object_name"] == name]
        .sort_values("t_sec")
        .reset_index(drop=True)
      )
      t = sub["t_sec"].to_numpy(dtype=float)
      on = sub["v2_interacting"].astype(bool).to_numpy()
      for a0, a1 in on_intervals(t, on):
        ax.barh(
          y,
          a1 - a0,
          left=a0,
          height=bar_height,
          facecolor="none",
          edgecolor="black",
          linewidth=dash_linewidth,
          linestyle="--",
          alpha=0.95,
        )

  ax.set_yticks(list(y_positions.values()))
  ax.set_yticklabels(list(y_positions.keys()), fontsize=object_fontsize)
  ax.set_ylim(-0.6, len(active) - 0.4)
  ax.set_ylabel(hand_label, fontsize=object_fontsize)
  ax.set_title(
    f"{hand_label} — fill={fill_label}, dashed={dashed_label}",
    fontsize=title_fontsize,
  )


def plot_teacher_student_hoi_figure(
  student_pair_df: pd.DataFrame,
  teacher_pair_df: pd.DataFrame,
  *,
  tick_sec: float = DEFAULT_TICK_SEC,
  output_path: Path | None = None,
  hands: tuple[str, ...] = ("left", "right"),
  object_order: list[str] | None = None,
  title: str = "Student vs teacher HOI (y_clean)",
  fill_label: str = "student",
  dashed_label: str = "teacher",
  t_min: float | None = None,
  t_max: float | None = None,
  object_fontsize: float = 9,
  time_labelsize: float = 8,
  title_fontsize: float = 12,
  panel_title_fontsize: float = 11,
  figsize: tuple[float, float] | None = None,
  bar_height: float = 0.7,
  dash_linewidth: float = 1.2,
  dpi: int = 150,
) -> plt.Figure:
  if "t_sec" not in student_pair_df.columns or "t_sec" not in teacher_pair_df.columns:
    raise KeyError("pair dfs need t_sec")

  data_t_min = min(float(student_pair_df["t_sec"].min()), float(teacher_pair_df["t_sec"].min()))
  data_t_max = max(float(student_pair_df["t_sec"].max()), float(teacher_pair_df["t_sec"].max()))
  display_t_min = data_t_min if t_min is None else float(t_min)
  display_t_max = data_t_max if t_max is None else float(t_max)
  window = (display_t_min, display_t_max) if (t_min is not None or t_max is not None) else None
  order = _object_order_from_pair_dfs(student_pair_df, teacher_pair_df, object_order=object_order)

  if figsize is None:
    figsize = (14, 3.4 * len(hands))
  fig, axes = plt.subplots(len(hands), 1, figsize=figsize, sharex=True)
  if len(hands) == 1:
    axes = [axes]

  for ax, hand in zip(axes, hands):
    if hand == "merged":
      s_hand = merge_hands_pair_df(student_pair_df)
      t_hand = merge_hands_pair_df(teacher_pair_df)
      hand_label = "both hands (L∪R)"
    else:
      s_hand = student_pair_df[student_pair_df["hand_side"] == hand]
      t_hand = teacher_pair_df[teacher_pair_df["hand_side"] == hand]
      hand_label = f"{hand} hand"
    plot_teacher_student_hoi_panel(
      ax,
      s_hand,
      t_hand,
      hand_label=hand_label,
      object_order=order,
      fill_label=fill_label,
      dashed_label=dashed_label,
      object_fontsize=object_fontsize,
      title_fontsize=panel_title_fontsize,
      window=window,
      bar_height=bar_height,
      dash_linewidth=dash_linewidth,
    )

  apply_timeline_time_axes(
    axes,
    display_t_min,
    display_t_max,
    tick_sec=tick_sec,
    time_labelsize=time_labelsize,
    xlabel_fontsize=max(time_labelsize, 10),
  )
  fig.suptitle(title, fontsize=title_fontsize, y=1.02)
  fig.tight_layout()
  if output_path is not None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
  return fig


def render_hoi_timeline_image(
  student_pair_df: pd.DataFrame,
  teacher_pair_df: pd.DataFrame,
  *,
  t_min: float,
  t_max: float,
  t_cursor: float | None = None,
  tick_sec: float = DEFAULT_TICK_SEC,
  object_fontsize: float = 9,
  time_labelsize: float = 8,
  panel_title_fontsize: float = 11,
  figsize: tuple[float, float] = (8.5, 4.8),
  bar_height: float = 0.7,
  dash_linewidth: float = 1.2,
  dpi: int = 200,
  show_title: bool = False,
  cursor_color: str = "#E45756",
  cursor_linewidth: float = 2.5,
) -> np.ndarray:
  """
  Rasterize merged student/teacher swimlane to RGB uint8 (H, W, 3).

  Full [t_min, t_max] bars are drawn; optional vertical cursor at t_cursor.
  """
  from matplotlib.backends.backend_agg import FigureCanvasAgg

  fig = plt.figure(figsize=figsize, dpi=dpi)
  canvas = FigureCanvasAgg(fig)
  ax = fig.add_subplot(111)

  s_hand = merge_hands_pair_df(student_pair_df)
  t_hand = merge_hands_pair_df(teacher_pair_df)
  plot_teacher_student_hoi_panel(
    ax,
    s_hand,
    t_hand,
    hand_label="both hands (L∪R)",
    fill_label="student",
    dashed_label="teacher",
    object_fontsize=object_fontsize,
    title_fontsize=panel_title_fontsize,
    window=(t_min, t_max),
    bar_height=bar_height,
    dash_linewidth=dash_linewidth,
  )
  configure_timeline_xaxis(
    ax, t_min, t_max, tick_sec=tick_sec, show_labels=True, labelsize=time_labelsize
  )
  ax.set_xlabel("t (s from clip start)", fontsize=max(time_labelsize, 10))
  if show_title:
    ax.set_title("fill=student, dashed=teacher", fontsize=panel_title_fontsize)
  if t_cursor is not None:
    ax.axvline(
      float(t_cursor),
      color=cursor_color,
      linewidth=cursor_linewidth,
      zorder=20,
      linestyle="-",
    )
  fig.tight_layout()
  canvas.draw()
  rgba = np.asarray(canvas.buffer_rgba())
  plt.close(fig)
  return rgba[:, :, :3].copy()


def plot_per_object_hoi_figure(
  pair_df: pd.DataFrame,
  timeline_df: pd.DataFrame | None = None,
  *,
  tick_sec: float = DEFAULT_TICK_SEC,
  output_path: Path | None = None,
  hands: tuple[str, ...] = ("left", "right"),
  object_order: list[str] | None = None,
) -> plt.Figure:
  """Demo timeline: per-hand swimlanes of interacting objects."""
  if "t_sec" not in pair_df.columns:
    raise KeyError("pair_df needs t_sec; re-run gt scorer")

  t_min = float(pair_df["t_sec"].min())
  t_max = float(pair_df["t_sec"].max())
  if timeline_df is not None and not timeline_df.empty:
    t_min = min(t_min, float(timeline_df["t_sec"].min()))
    t_max = max(t_max, float(timeline_df["t_sec"].max()))

  # Prefer a stable demo order, then append any other objects present in the data
  # (e.g. whiteboard_* on non-desk clips). Never silently drop teacher ON lanes.
  preferred = [
    "keyboard",
    "mouse",
    "mug_white",
    "coffee_pot",
    "puzzle_toy",
    "cellphone",
    "whiteboard_marker",
    "whiteboard_eraser",
  ]
  if object_order is not None:
    order = list(object_order)
  else:
    present = sorted({str(n) for n in pair_df["object_name"].unique()})
    if timeline_df is not None and not timeline_df.empty and "gt_object_name" in timeline_df.columns:
      present.extend(str(n) for n in timeline_df["gt_object_name"].unique() if str(n) != "none")
      present = sorted(set(present))
    order = [n for n in preferred if n in present]
    order.extend(n for n in present if n not in order)

  fig, axes = plt.subplots(len(hands), 1, figsize=(14, 3.2 * len(hands)), sharex=True)
  if len(hands) == 1:
    axes = [axes]

  for ax, hand in zip(axes, hands):
    hand_pairs = pair_df[pair_df["hand_side"] == hand]
    gt_hand = None
    if timeline_df is not None and not timeline_df.empty:
      gt_hand = hand_timeline_df(timeline_df, hand)
    plot_per_object_hoi_panel(
      ax,
      hand_pairs,
      gt_hand,
      hand_label=f"{hand} hand",
      object_order=order,
    )

  apply_timeline_time_axes(axes, t_min, t_max, tick_sec=tick_sec)
  fig.suptitle("Per-hand · per-object HOI (gt_world teacher)", fontsize=12, y=1.01)
  fig.tight_layout()
  if output_path is not None:
    _save_figure(fig, output_path)
  return fig
