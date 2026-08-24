"""Post-process teacher pair scores into a cleaner HOI timeline.

Locked order (no GT used):
  1) Merge short OFF gaps within the same (hand, object) predicted ON stream
  2) Drop remaining micro ON bars; snapshot survivors as protected
  3) Gate-conditioned dwell-onset backfill (frame_ok already true)
  4) Resolve same-hand multi-object overlap (sticky; prefer moving / co-motion)
  5) Protected remicro: drop short bars unless they are remnants of protected parents
  6) Drop lone short spikes: short duration with no same-object neighbor within N seconds
  7) 1D binary closing then opening per (hand, object) lane
  8) Re-run sticky exclusive (closing must not leave multi-ON)

Note: post-exclusive same-object re-merge was tried on P0003 and hurt seg_f1
(over-kept junk bars); omitted from the default stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lib.eval_timeline import TimelineSegment, pred_segments_from_pair_scores


@dataclass(frozen=True)
class TimelineCleanupConfig:
  # P0003: gaps ≥~0.15s over-merge across real put-down/pick-up and hurt seg_rec.
  max_gap_sec: float = 0.12
  min_on_sec: float = 0.20
  # Walk back from dwell onset while spatial/motion gate already held.
  backfill_max_sec: float = 0.18
  # Remicro floor for remnants of pre-exclusive ≥ min_on segments.
  # 0.18 tuned on P0003 (0.08 keeps phone fragments that never hit tIoU≥0.5 and hurt seg_f1).
  remicro_protected_min_on_sec: float = 0.18
  # Final pass: drop short bars isolated on both sides (same hand+object).
  lone_short_max_duration_sec: float = 1.20
  lone_neighbor_min_gap_sec: float = 3.0
  # Final morph: close fills thin OFF dropouts; open erases short ON flickers.
  # 0.10/0.35 promoted on P0003 (0.15/0.30 dropped s_prec vs post-lone-short).
  morph_close_sec: float = 0.10
  morph_open_sec: float = 0.35


def merge_same_object_gaps(
  segments: list[TimelineSegment],
  max_gap_sec: float,
) -> list[TimelineSegment]:
  """Merge consecutive same-(hand, object) segments separated by a short gap."""
  if not segments:
    return []
  by_key: dict[tuple[str, str], list[TimelineSegment]] = {}
  for s in segments:
    by_key.setdefault((s.hand_side, s.object_name), []).append(s)

  out: list[TimelineSegment] = []
  for _, group in by_key.items():
    group = sorted(group, key=lambda s: (s.t_start, s.t_end))
    cur = group[0]
    for nxt in group[1:]:
      gap = nxt.t_start - cur.t_end
      if gap <= max_gap_sec:
        cur = TimelineSegment(cur.hand_side, cur.object_name, cur.t_start, max(cur.t_end, nxt.t_end))
      else:
        out.append(cur)
        cur = nxt
    out.append(cur)
  return out


def drop_micro_segments(
  segments: list[TimelineSegment],
  min_on_sec: float,
) -> list[TimelineSegment]:
  return [s for s in segments if s.duration >= min_on_sec]


def _segments_overlap(a: TimelineSegment, b: TimelineSegment) -> bool:
  """Half-open overlap: [t_start, t_end) intervals overlap if start < other.end and end > other.start."""
  return a.t_start < b.t_end and a.t_end > b.t_start


def is_protected_remnant(
  segment: TimelineSegment,
  protected: list[TimelineSegment],
) -> bool:
  """True if segment overlaps a protected parent of the same (hand, object)."""
  for p in protected:
    if (
      p.hand_side == segment.hand_side
      and p.object_name == segment.object_name
      and _segments_overlap(segment, p)
    ):
      return True
  return False


def drop_micro_segments_protected(
  segments: list[TimelineSegment],
  protected: list[TimelineSegment],
  min_on_sec: float,
  protected_floor: float,
) -> tuple[list[TimelineSegment], int]:
  """
  Remicro drop with protected exemption.

  Drop S if duration < min_on_sec unless S overlaps a protected parent
  (same hand/object). Protected remnants still drop if duration < protected_floor.
  Returns (kept_segments, n_kept_because_protected).
  """
  kept: list[TimelineSegment] = []
  n_kept_protected = 0
  for s in segments:
    if s.duration >= min_on_sec:
      kept.append(s)
      continue
    if is_protected_remnant(s, protected) and s.duration >= protected_floor:
      kept.append(s)
      n_kept_protected += 1
      continue
    # Drop: too short, and either unprotected or below protected floor.
  return kept, n_kept_protected


def drop_lone_short_segments(
  segments: list[TimelineSegment],
  *,
  max_duration_sec: float,
  neighbor_min_gap_sec: float,
) -> tuple[list[TimelineSegment], int]:
  """
  Drop short segments with no same-(hand, object) neighbor within ``neighbor_min_gap_sec``.

  A segment is lone when gaps to both the previous and next segment (if any) exceed
  the threshold. Keeps clustered short bars (e.g. right mouse ~101–107s) while removing
  isolated keyboard spikes.
  """
  if max_duration_sec <= 0 or neighbor_min_gap_sec <= 0:
    return list(segments), 0

  by_lane: dict[tuple[str, str], list[TimelineSegment]] = {}
  for s in segments:
    by_lane.setdefault((s.hand_side, s.object_name), []).append(s)

  kept: list[TimelineSegment] = []
  n_dropped = 0
  for _, group in by_lane.items():
    group = sorted(group, key=lambda s: (s.t_start, s.t_end))
    for i, s in enumerate(group):
      if s.duration > max_duration_sec:
        kept.append(s)
        continue
      gap_before = s.t_start - group[i - 1].t_end if i > 0 else float("inf")
      gap_after = group[i + 1].t_start - s.t_end if i + 1 < len(group) else float("inf")
      if len(group) == 1:
        kept.append(s)
        continue
      if i == 0:
        is_lone = gap_after > neighbor_min_gap_sec
      elif i == len(group) - 1:
        is_lone = gap_before > neighbor_min_gap_sec
      else:
        is_lone = (
          gap_before > neighbor_min_gap_sec and gap_after > neighbor_min_gap_sec
        )
      if is_lone:
        n_dropped += 1
        continue
      kept.append(s)
  return kept, n_dropped


def _median_dt_sec(t_sec: np.ndarray) -> float:
  if len(t_sec) <= 1:
    return 1.0 / 30.0
  diffs = np.diff(np.sort(t_sec.astype(float)))
  positive = diffs[diffs > 0]
  return float(np.median(positive)) if len(positive) else 1.0 / 30.0


def _morph_radius_samples(radius_sec: float, median_dt: float) -> int:
  if radius_sec <= 0:
    return 0
  return max(1, int(round(radius_sec / max(median_dt, 1e-6))))


def _binary_dilate_1d(on: np.ndarray, radius: int) -> np.ndarray:
  if radius <= 0 or len(on) == 0:
    return on.astype(bool).copy()
  x = on.astype(bool)
  pad = np.pad(x, (radius, radius), constant_values=False)
  out = np.empty(len(x), dtype=bool)
  for i in range(len(x)):
    out[i] = pad[i : i + 2 * radius + 1].any()
  return out


def _binary_erode_1d(on: np.ndarray, radius: int) -> np.ndarray:
  if radius <= 0 or len(on) == 0:
    return on.astype(bool).copy()
  x = on.astype(bool)
  pad = np.pad(x, (radius, radius), constant_values=True)
  out = np.empty(len(x), dtype=bool)
  for i in range(len(x)):
    out[i] = pad[i : i + 2 * radius + 1].all()
  return out


def _binary_closing_1d(on: np.ndarray, radius: int) -> np.ndarray:
  if radius <= 0:
    return on.astype(bool).copy()
  return _binary_erode_1d(_binary_dilate_1d(on, radius), radius)


def _binary_opening_1d(on: np.ndarray, radius: int) -> np.ndarray:
  if radius <= 0:
    return on.astype(bool).copy()
  return _binary_dilate_1d(_binary_erode_1d(on, radius), radius)


def morph_close_open_pair_scores(
  pair_df: pd.DataFrame,
  close_sec: float,
  open_sec: float,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
  """
  Per-(hand, object) 1D binary closing then opening on ``v2_interacting``.

  Operates only on existing lane rows; does not invent missing frames.
  """
  if close_sec <= 0 and open_sec <= 0:
    return pair_df.copy(), {
      "morph_close_sec": close_sec,
      "morph_open_sec": open_sec,
      "n_morph_on_added": 0,
      "n_morph_on_removed": 0,
    }

  out = pair_df.copy()
  before = out["v2_interacting"].astype(bool).to_numpy().copy()
  close_r = 0
  open_r = 0

  for (_, _), idxs_arr in out.groupby(["hand_side", "object_name"], sort=False).groups.items():
    idxs = np.asarray(list(idxs_arr), dtype=int)
    order = np.argsort(out.loc[idxs, "t_sec"].to_numpy(dtype=float))
    idxs = idxs[order]
    t = out.loc[idxs, "t_sec"].to_numpy(dtype=float)
    on = out.loc[idxs, "v2_interacting"].astype(bool).to_numpy()
    median_dt = _median_dt_sec(t)
    if close_sec > 0:
      close_r = _morph_radius_samples(close_sec, median_dt)
      on = _binary_closing_1d(on, close_r)
    if open_sec > 0:
      open_r = _morph_radius_samples(open_sec, median_dt)
      on = _binary_opening_1d(on, open_r)
    out.loc[idxs, "v2_interacting"] = on.astype(int)

  after = out["v2_interacting"].astype(bool).to_numpy()
  return out, {
    "morph_close_sec": close_sec,
    "morph_open_sec": open_sec,
    "morph_close_radius_samples": close_r,
    "morph_open_radius_samples": open_r,
    "n_morph_on_added": int((~before & after).sum()),
    "n_morph_on_removed": int((before & ~after).sum()),
  }


def _apply_segment_mask_to_pair_df(
  pair_df: pd.DataFrame,
  segments: list[TimelineSegment],
) -> pd.DataFrame:
  """Set v2_interacting from allowed segments only (per hand, object)."""
  out = pair_df.copy()
  segs_by_lane: dict[tuple[str, str], list[TimelineSegment]] = {}
  for s in segments:
    segs_by_lane.setdefault((s.hand_side, s.object_name), []).append(s)

  interacting = np.zeros(len(out), dtype=bool)
  for (hand, obj), lane_segs in segs_by_lane.items():
    mask = (out["hand_side"].to_numpy() == hand) & (out["object_name"].to_numpy() == obj)
    idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
      continue
    t = out["t_sec"].to_numpy(dtype=float)[idxs]
    on = np.zeros(len(t), dtype=bool)
    for s in lane_segs:
      # Half-open [t_start, t_end), same as on_intervals / swimlane fills.
      on |= (t >= s.t_start) & (t < s.t_end)
    interacting[idxs] = on
  out["v2_interacting"] = interacting.astype(int)
  return out


def _gate_ok_series(sub: pd.DataFrame) -> np.ndarray:
  """Prefer explicit frame/motion gate columns; never invent ON without a gate."""
  if "v2_frame_ok" in sub.columns:
    return sub["v2_frame_ok"].astype(bool).to_numpy()
  if "v2_motion_ok" in sub.columns:
    return sub["v2_motion_ok"].astype(bool).to_numpy()
  return np.zeros(len(sub), dtype=bool)


def backfill_dwell_onset(
  pair_df: pd.DataFrame,
  backfill_max_sec: float,
) -> pd.DataFrame:
  """
  Extend each interacting onset backward while the gate was already true.

  Cap: do not walk more than ``backfill_max_sec`` before the first ON sample.
  Does not resolve same-hand conflicts — call exclusive afterward.
  """
  if backfill_max_sec <= 0:
    return pair_df.copy()

  out = pair_df.copy()
  interacting = out["v2_interacting"].astype(bool).to_numpy().copy()

  for (_, _), idxs_arr in out.groupby(["hand_side", "object_name"], sort=False).groups.items():
    idxs = np.asarray(list(idxs_arr), dtype=int)
    order = np.argsort(out.loc[idxs, "t_sec"].to_numpy(dtype=float))
    idxs = idxs[order]
    t = out.loc[idxs, "t_sec"].to_numpy(dtype=float)
    on = interacting[idxs]
    gate = _gate_ok_series(out.loc[idxs])

    i = 0
    n = len(idxs)
    while i < n:
      if not on[i]:
        i += 1
        continue
      onset_t = float(t[i])
      j = i - 1
      while j >= 0 and (onset_t - float(t[j])) <= backfill_max_sec and bool(gate[j]) and not bool(on[j]):
        interacting[idxs[j]] = True
        j -= 1
      while i < n and on[i]:
        i += 1

  out["v2_interacting"] = interacting.astype(int)
  return out


def _winner_key(row: pd.Series) -> tuple:
  """Higher is better. Prefer co-motion over stationary, then nearer, then dwell/iou."""
  co_motion = bool(row.get("v2_co_motion_ok", 0))
  stationary = bool(row.get("v2_stationary_ok", 0))
  slow_manip = bool(row.get("v2_slow_manip_ok", 0))
  move_rank = 2 if co_motion else (1 if (stationary or slow_manip) else 0)
  aabb = row.get("v2_aabb_dist_m", np.nan)
  aabb = float(aabb) if pd.notna(aabb) else 1e9
  dwell = float(row.get("v2_dwell_ms", 0.0) or 0.0)
  iou = float(row.get("iou", 0.0) or 0.0)
  return (move_rank, -aabb, dwell, iou)


def resolve_same_hand_overlap(pair_df: pd.DataFrame) -> pd.DataFrame:
  """
  Keep at most one interacting object per (timestamp, hand).

  Sticky hysteresis: keep the current object while it remains ON; only switch
  when it drops out. Avoids A/B/A/B frame flicker that fragments the timeline.
  When starting fresh, prefer co-motion / nearer AABB via `_winner_key`.
  """
  out = pair_df.copy()
  on_mask = out["v2_interacting"].astype(bool)
  if not on_mask.any():
    return out

  new_on = pd.Series(False, index=out.index)
  for _, h_all in out.groupby("hand_side", sort=False):
    ts_order = (
      h_all.sort_values(["t_sec", "timestamp_ns"])["timestamp_ns"]
      .drop_duplicates()
      .tolist()
    )
    current: str | None = None
    for ts in ts_order:
      g = h_all[h_all["timestamp_ns"] == ts]
      cands = g[g["v2_interacting"].astype(bool)]
      if cands.empty:
        current = None
        continue
      names = set(cands["object_name"].astype(str))
      if current is not None and current in names:
        pick = int(cands[cands["object_name"].astype(str) == current].index[0])
      else:
        pick = max(cands.index.tolist(), key=lambda i: _winner_key(out.loc[i]))
        current = str(out.loc[pick, "object_name"])
      new_on.loc[pick] = True

  out["v2_interacting"] = new_on.astype(int)
  return out


def cleanup_pair_scores(
  pair_df: pd.DataFrame,
  config: TimelineCleanupConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
  """
  Apply merge → drop → gated backfill → exclusive → protected remicro.
  """
  config = config or TimelineCleanupConfig()
  raw_segments = pred_segments_from_pair_scores(pair_df)
  merged = merge_same_object_gaps(raw_segments, config.max_gap_sec)
  pruned = drop_micro_segments(merged, config.min_on_sec)
  protected = list(pruned)
  mid = _apply_segment_mask_to_pair_df(pair_df, pruned)
  filled = backfill_dwell_onset(mid, config.backfill_max_sec)
  exclusive = resolve_same_hand_overlap(filled)
  after_excl = pred_segments_from_pair_scores(exclusive)
  final_segs, n_kept_protected = drop_micro_segments_protected(
    after_excl,
    protected,
    config.min_on_sec,
    config.remicro_protected_min_on_sec,
  )
  lone_dropped_segs, n_lone_dropped = drop_lone_short_segments(
    final_segs,
    max_duration_sec=config.lone_short_max_duration_sec,
    neighbor_min_gap_sec=config.lone_neighbor_min_gap_sec,
  )
  after_lone = _apply_segment_mask_to_pair_df(exclusive, lone_dropped_segs)
  morphed, morph_stats = morph_close_open_pair_scores(
    after_lone,
    config.morph_close_sec,
    config.morph_open_sec,
  )
  clean = resolve_same_hand_overlap(morphed)
  final_segs = pred_segments_from_pair_scores(clean)
  multi = (
    clean[clean["v2_interacting"].astype(bool)]
    .groupby(["timestamp_ns", "hand_side"])
    .size()
  )
  n_multi = int((multi > 1).sum()) if len(multi) else 0
  stats = {
    "max_gap_sec": config.max_gap_sec,
    "min_on_sec": config.min_on_sec,
    "backfill_max_sec": config.backfill_max_sec,
    "remicro_protected_min_on_sec": config.remicro_protected_min_on_sec,
    "lone_short_max_duration_sec": config.lone_short_max_duration_sec,
    "lone_neighbor_min_gap_sec": config.lone_neighbor_min_gap_sec,
    "morph_close_sec": config.morph_close_sec,
    "morph_open_sec": config.morph_open_sec,
    "n_segments_raw": len(raw_segments),
    "n_segments_merged": len(merged),
    "n_segments_after_drop": len(pruned),
    "n_protected": len(protected),
    "n_segments_after_exclusive": len(after_excl),
    "n_segments_after_lone_short": len(lone_dropped_segs),
    "n_segments_clean": len(final_segs),
    "n_remicro_kept_protected": n_kept_protected,
    "n_lone_short_dropped": n_lone_dropped,
    "n_morph_on_added": morph_stats["n_morph_on_added"],
    "n_morph_on_removed": morph_stats["n_morph_on_removed"],
    "n_pair_on_raw": int(pair_df["v2_interacting"].astype(bool).sum()),
    "n_pair_on_clean": int(clean["v2_interacting"].astype(bool).sum()),
    "n_multi_on_frames": n_multi,
  }
  return clean, stats


def frame_scores_from_clean_pairs(pair_clean: pd.DataFrame) -> pd.DataFrame:
  """One row per (timestamp, hand): leader among interacting, else none."""
  rows = []
  for (ts, hand), group in pair_clean.groupby(["timestamp_ns", "hand_side"], sort=False):
    on = group[group["v2_interacting"].astype(bool)]
    t_sec = float(group["t_sec"].iloc[0])
    frame_index = int(group["frame_index"].iloc[0]) if "frame_index" in group.columns else -1
    if on.empty:
      rows.append(
        {
          "timestamp_ns": int(ts),
          "t_sec": t_sec,
          "frame_index": frame_index,
          "hand_side": str(hand),
          "v2_interacting": 0,
          "v2_object_name": "none",
          "v2_dwell_ms": 0.0,
          "v2_aabb_dist_m": float("nan"),
          "iou": 0.0,
          "v2_co_motion_ok": 0,
          "v2_stationary_ok": 0,
        }
      )
      continue
    r = on.iloc[0]
    rows.append(
      {
        "timestamp_ns": int(ts),
        "t_sec": t_sec,
        "frame_index": frame_index,
        "hand_side": str(hand),
        "v2_interacting": 1,
        "v2_object_name": str(r["object_name"]),
        "v2_dwell_ms": float(r.get("v2_dwell_ms", 0.0) or 0.0),
        "v2_aabb_dist_m": float(r["v2_aabb_dist_m"]) if pd.notna(r.get("v2_aabb_dist_m")) else float("nan"),
        "iou": float(r.get("iou", 0.0) or 0.0),
        "v2_co_motion_ok": int(r.get("v2_co_motion_ok", 0) or 0),
        "v2_stationary_ok": int(r.get("v2_stationary_ok", 0) or 0),
      }
    )
  return pd.DataFrame(rows)


def segments_dataframe(segments: list[TimelineSegment]) -> pd.DataFrame:
  return pd.DataFrame(
    [
      {
        "hand_side": s.hand_side,
        "object_name": s.object_name,
        "t_start": s.t_start,
        "t_end": s.t_end,
        "duration_sec": s.duration,
      }
      for s in sorted(segments, key=lambda x: (x.hand_side, x.t_start, x.object_name))
    ]
  )
