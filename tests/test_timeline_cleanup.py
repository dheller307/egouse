"""Unit tests for timeline cleanup (merge → drop → exclusive)."""

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
import numpy as np

from lib.eval_timeline import TimelineSegment, pred_segments_from_pair_scores
from lib.scoring.timeline_cleanup import (
  TimelineCleanupConfig,
  _binary_closing_1d,
  _binary_opening_1d,
  backfill_dwell_onset,
  cleanup_pair_scores,
  drop_lone_short_segments,
  drop_micro_segments,
  drop_micro_segments_protected,
  merge_same_object_gaps,
  morph_close_open_pair_scores,
  resolve_same_hand_overlap,
)


def _lane_pair_df(hand: str, obj: str, t_sec: list[float], on: list[int]) -> pd.DataFrame:
  n = len(t_sec)
  return pd.DataFrame(
    {
      "timestamp_ns": list(range(n)),
      "t_sec": t_sec,
      "frame_index": list(range(n)),
      "hand_side": [hand] * n,
      "object_name": [obj] * n,
      "v2_interacting": on,
      "v2_frame_ok": on,
      "v2_co_motion_ok": on,
      "v2_stationary_ok": [0] * n,
      "v2_aabb_dist_m": [0.05] * n,
      "v2_dwell_ms": [300.0] * n,
      "iou": [0.4] * n,
    }
  )


class TestMergeGaps(unittest.TestCase):
  def test_merges_short_gap(self):
    segs = [
      TimelineSegment("left", "keyboard", 0.0, 1.0),
      TimelineSegment("left", "keyboard", 1.15, 2.0),
    ]
    out = merge_same_object_gaps(segs, max_gap_sec=0.25)
    self.assertEqual(len(out), 1)
    self.assertAlmostEqual(out[0].t_start, 0.0)
    self.assertAlmostEqual(out[0].t_end, 2.0)

  def test_does_not_merge_across_objects(self):
    segs = [
      TimelineSegment("left", "keyboard", 0.0, 1.0),
      TimelineSegment("left", "mouse", 1.05, 2.0),
    ]
    out = merge_same_object_gaps(segs, max_gap_sec=0.25)
    self.assertEqual(len(out), 2)

  def test_does_not_merge_large_gap(self):
    segs = [
      TimelineSegment("right", "mug", 0.0, 1.0),
      TimelineSegment("right", "mug", 1.5, 2.0),
    ]
    out = merge_same_object_gaps(segs, max_gap_sec=0.25)
    self.assertEqual(len(out), 2)


class TestDropMicro(unittest.TestCase):
  def test_drops_short(self):
    segs = [
      TimelineSegment("left", "a", 0.0, 0.1),
      TimelineSegment("left", "b", 1.0, 2.0),
    ]
    out = drop_micro_segments(segs, min_on_sec=0.2)
    self.assertEqual(len(out), 1)
    self.assertEqual(out[0].object_name, "b")


class TestExclusive(unittest.TestCase):
  def test_prefers_co_motion_on_fresh_start(self):
    pair = pd.DataFrame(
      {
        "timestamp_ns": [1, 1],
        "t_sec": [0.0, 0.0],
        "hand_side": ["left", "left"],
        "object_name": ["keyboard", "mouse"],
        "v2_interacting": [1, 1],
        "v2_co_motion_ok": [0, 1],
        "v2_stationary_ok": [1, 0],
        "v2_aabb_dist_m": [0.05, 0.10],
        "v2_dwell_ms": [500.0, 100.0],
        "iou": [0.5, 0.2],
      }
    )
    out = resolve_same_hand_overlap(pair)
    on = out[out["v2_interacting"].astype(bool)]
    self.assertEqual(len(on), 1)
    self.assertEqual(on.iloc[0]["object_name"], "mouse")

  def test_sticky_keeps_current_while_on(self):
    # Stationary keyboard is current; co-motion mouse appears while keyboard stays ON.
    pair = pd.DataFrame(
      {
        "timestamp_ns": [1, 1, 2, 2],
        "t_sec": [0.0, 0.0, 0.1, 0.1],
        "hand_side": ["left"] * 4,
        "object_name": ["keyboard", "mouse", "keyboard", "mouse"],
        "v2_interacting": [1, 0, 1, 1],
        "v2_co_motion_ok": [0, 0, 0, 1],
        "v2_stationary_ok": [1, 0, 1, 0],
        "v2_aabb_dist_m": [0.05, 0.10, 0.05, 0.10],
        "v2_dwell_ms": [500.0, 0.0, 500.0, 100.0],
        "iou": [0.5, 0.0, 0.5, 0.2],
      }
    )
    out = resolve_same_hand_overlap(pair)
    on = out[out["v2_interacting"].astype(bool)].sort_values("t_sec")
    self.assertEqual(list(on["object_name"]), ["keyboard", "keyboard"])


class TestCleanupPipeline(unittest.TestCase):
  def test_merge_then_keep_chain(self):
    # Two short ON runs of same object separated by one OFF frame; gap ~0.1s
    t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    on = [1, 1, 0, 1, 1, 0]
    pair = pd.DataFrame(
      {
        "timestamp_ns": list(range(len(t))),
        "t_sec": t,
        "frame_index": list(range(len(t))),
        "hand_side": ["left"] * len(t),
        "object_name": ["keyboard"] * len(t),
        "v2_interacting": on,
        "v2_frame_ok": on,
        "v2_co_motion_ok": on,
        "v2_stationary_ok": [0] * len(t),
        "v2_aabb_dist_m": [0.05] * len(t),
        "v2_dwell_ms": [300.0] * len(t),
        "iou": [0.4] * len(t),
      }
    )
    clean, stats = cleanup_pair_scores(
      pair, TimelineCleanupConfig(max_gap_sec=0.12, min_on_sec=0.20)
    )
    segs = pred_segments_from_pair_scores(clean)
    self.assertEqual(stats["n_segments_raw"], 2)
    self.assertEqual(len(segs), 1)
    self.assertEqual(int(clean.loc[clean["t_sec"] == 0.2, "v2_interacting"].iloc[0]), 1)
    self.assertEqual(stats["n_multi_on_frames"], 0)


class TestBackfill(unittest.TestCase):
  def test_backfills_gate_prefix_only(self):
    # Gate true for 0.0–0.15; interacting starts at 0.1 (dwell lag).
    pair = pd.DataFrame(
      {
        "timestamp_ns": [0, 1, 2, 3, 4],
        "t_sec": [0.0, 0.05, 0.10, 0.15, 0.20],
        "hand_side": ["left"] * 5,
        "object_name": ["mug_white"] * 5,
        "v2_interacting": [0, 0, 1, 1, 1],
        "v2_frame_ok": [1, 1, 1, 1, 1],
        "v2_co_motion_ok": [0] * 5,
        "v2_stationary_ok": [1] * 5,
        "v2_aabb_dist_m": [0.05] * 5,
        "v2_dwell_ms": [0.0, 50.0, 100.0, 150.0, 200.0],
        "iou": [0.2] * 5,
      }
    )
    out = backfill_dwell_onset(pair, backfill_max_sec=0.18)
    self.assertEqual(list(out["v2_interacting"]), [1, 1, 1, 1, 1])

  def test_no_backfill_without_gate(self):
    pair = pd.DataFrame(
      {
        "timestamp_ns": [0, 1, 2],
        "t_sec": [0.0, 0.05, 0.10],
        "hand_side": ["left"] * 3,
        "object_name": ["mug_white"] * 3,
        "v2_interacting": [0, 0, 1],
        "v2_frame_ok": [0, 0, 1],
        "v2_co_motion_ok": [0] * 3,
        "v2_stationary_ok": [0] * 3,
        "v2_aabb_dist_m": [0.2] * 3,
        "v2_dwell_ms": [0.0, 0.0, 100.0],
        "iou": [0.0] * 3,
      }
    )
    out = backfill_dwell_onset(pair, backfill_max_sec=0.18)
    self.assertEqual(list(out["v2_interacting"]), [0, 0, 1])

  def test_cleanup_exclusive_after_backfill(self):
    # Two objects both gated; keyboard interacts after dwell; mouse also gate-true
    # surrounding keyboard onset — exclusive must leave ≤1 ON per frame.
    rows = []
    for i, t in enumerate([0.0, 0.05, 0.10, 0.15, 0.20]):
      for obj, on in (("keyboard", int(t >= 0.10)), ("mouse", 0)):
        rows.append(
          {
            "timestamp_ns": i,
            "t_sec": t,
            "frame_index": i,
            "hand_side": "right",
            "object_name": obj,
            "v2_interacting": on,
            "v2_frame_ok": 1,
            "v2_motion_ok": 1,
            "v2_co_motion_ok": 0,
            "v2_stationary_ok": 1,
            "v2_aabb_dist_m": 0.04 if obj == "keyboard" else 0.08,
            "v2_dwell_ms": 200.0 if on else 80.0,
            "iou": 0.3,
          }
        )
    pair = pd.DataFrame(rows)
    clean, stats = cleanup_pair_scores(
      pair, TimelineCleanupConfig(max_gap_sec=0.12, min_on_sec=0.05, backfill_max_sec=0.18)
    )
    multi = (
      clean[clean["v2_interacting"].astype(bool)]
      .groupby(["timestamp_ns", "hand_side"])
      .size()
    )
    self.assertEqual(int((multi > 1).sum()), 0)
    self.assertEqual(stats["n_multi_on_frames"], 0)


class TestProtectedRemicro(unittest.TestCase):
  def test_keeps_protected_remnant_below_min_on(self):
    # Exclusive truncated a ≥0.20s bar to a 0.12s remnant → keep via protection.
    protected = [TimelineSegment("right", "cellphone", 0.0, 0.5)]
    segs = [TimelineSegment("right", "cellphone", 0.0, 0.12)]
    kept, n_prot = drop_micro_segments_protected(
      segs, protected, min_on_sec=0.20, protected_floor=0.08
    )
    self.assertEqual(len(kept), 1)
    self.assertEqual(n_prot, 1)

  def test_drops_unprotected_spike(self):
    # Fresh exclusive spike with no protected parent → drop.
    protected = [TimelineSegment("right", "keyboard", 0.0, 1.0)]
    segs = [TimelineSegment("right", "mouse", 0.5, 0.60)]
    kept, n_prot = drop_micro_segments_protected(
      segs, protected, min_on_sec=0.20, protected_floor=0.08
    )
    self.assertEqual(len(kept), 0)
    self.assertEqual(n_prot, 0)

  def test_drops_protected_below_floor(self):
    protected = [TimelineSegment("left", "cellphone", 0.0, 0.5)]
    segs = [TimelineSegment("left", "cellphone", 0.0, 0.05)]
    kept, n_prot = drop_micro_segments_protected(
      segs, protected, min_on_sec=0.20, protected_floor=0.08
    )
    self.assertEqual(len(kept), 0)
    self.assertEqual(n_prot, 0)

  def test_post_exclusive_merge_reconnects(self):
    segs = [
      TimelineSegment("left", "mug", 0.0, 0.5),
      TimelineSegment("left", "mug", 0.58, 1.0),
    ]
    out = merge_same_object_gaps(segs, max_gap_sec=0.12)
    self.assertEqual(len(out), 1)
    self.assertAlmostEqual(out[0].t_start, 0.0)
    self.assertAlmostEqual(out[0].t_end, 1.0)

  def test_pipeline_keeps_protected_after_exclusive_trim(self):
    # Long cellphone ON; sticky exclusive loses middle frames to keyboard then
    # cellphone remnant is short of min_on but protected → kept.
    rows = []
    for i, t in enumerate([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]):
      phone_on = 1 if t < 0.20 else 0
      # Keyboard briefly wins exclusive mid-window via co-motion; phone still
      # has a long pre-exclusive bar that becomes protected.
      key_on = 1 if 0.10 <= t < 0.20 else 0
      for obj, on, cm, aabb in (
        ("cellphone", phone_on, 0, 0.05),
        ("keyboard", key_on, 1, 0.04),
      ):
        rows.append(
          {
            "timestamp_ns": i,
            "t_sec": t,
            "frame_index": i,
            "hand_side": "right",
            "object_name": obj,
            "v2_interacting": on,
            "v2_frame_ok": 1,
            "v2_co_motion_ok": cm,
            "v2_stationary_ok": 1 - cm,
            "v2_aabb_dist_m": aabb,
            "v2_dwell_ms": 300.0 if on else 0.0,
            "iou": 0.3,
          }
        )
    pair = pd.DataFrame(rows)
    clean, stats = cleanup_pair_scores(
      pair,
      TimelineCleanupConfig(
        max_gap_sec=0.12,
        min_on_sec=0.20,
        backfill_max_sec=0.0,
        remicro_protected_min_on_sec=0.08,
        morph_close_sec=0.0,
        morph_open_sec=0.0,
      ),
    )
    self.assertEqual(stats["n_multi_on_frames"], 0)
    phone_on = clean[
      (clean["object_name"] == "cellphone") & clean["v2_interacting"].astype(bool)
    ]
    # Phone should retain early frames (protected remnant), not be wiped entirely.
    self.assertGreater(len(phone_on), 0)


class TestLoneShortDrop(unittest.TestCase):
  def test_drops_isolated_keyboard_spike(self):
    segs = [
      TimelineSegment("right", "keyboard", 1.5, 2.8),
      TimelineSegment("right", "keyboard", 12.4, 13.2),
      TimelineSegment("right", "keyboard", 44.8, 45.7),
    ]
    kept, n = drop_lone_short_segments(
      segs, max_duration_sec=1.20, neighbor_min_gap_sec=3.0
    )
    self.assertEqual(n, 2)
    self.assertEqual(len(kept), 1)
    self.assertAlmostEqual(kept[0].t_start, 1.5)

  def test_keeps_clustered_short_mouse(self):
    segs = [
      TimelineSegment("right", "mouse", 101.7, 102.6),
      TimelineSegment("right", "mouse", 104.1, 104.9),
      TimelineSegment("right", "mouse", 105.6, 106.0),
      TimelineSegment("right", "mouse", 107.4, 109.1),
    ]
    kept, n = drop_lone_short_segments(
      segs, max_duration_sec=1.20, neighbor_min_gap_sec=3.0
    )
    self.assertEqual(n, 0)
    self.assertEqual(len(kept), 4)

  def test_drops_left_keyboard_isolated_blip(self):
    segs = [
      TimelineSegment("left", "keyboard", 60.0, 62.9),
      TimelineSegment("left", "keyboard", 88.3, 88.8),
      TimelineSegment("left", "keyboard", 98.5, 99.9),
    ]
    kept, n = drop_lone_short_segments(
      segs, max_duration_sec=1.20, neighbor_min_gap_sec=3.0
    )
    self.assertEqual(n, 1)
    self.assertEqual(len(kept), 2)
    starts = [s.t_start for s in kept]
    self.assertNotIn(88.3, starts)


class TestMorphCloseOpen(unittest.TestCase):
  def test_close_fills_thin_dropout(self):
    dt = 0.033
    t = [i * dt for i in range(8)]
    on = [1, 1, 1, 0, 1, 1, 1, 1]
    out = _binary_closing_1d(np.array(on, dtype=bool), radius=2)
    self.assertTrue(out.all())

  def test_open_erases_short_pulse(self):
    dt = 0.033
    t = [i * dt for i in range(10)]
    on = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    out = _binary_opening_1d(np.array(on, dtype=bool), radius=5)
    self.assertFalse(out.any())

  def test_open_keeps_longer_pulse(self):
    dt = 0.033
    on = [0] * 3 + [1] * 15 + [0] * 3
    out = _binary_opening_1d(np.array(on, dtype=bool), radius=5)
    self.assertTrue(out[5:16].all())

  def test_close_open_merges_fragment_and_kills_flicker(self):
    dt = 0.033
    t = [i * dt for i in range(30)]
    on = [0] * 5 + [1] * 4 + [0] + [1] * 4 + [0] * 16
    pair = _lane_pair_df("left", "puzzle_toy", t, on)
    morphed, stats = morph_close_open_pair_scores(pair, close_sec=0.15, open_sec=0.30)
    segs = pred_segments_from_pair_scores(morphed)
    self.assertEqual(len(segs), 1)
    self.assertGreater(stats["n_morph_on_added"], 0)

    flicker_t = [i * dt for i in range(12)]
    flicker_on = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    flicker = _lane_pair_df("right", "mug_white", flicker_t, flicker_on)
    cleaned, stats2 = morph_close_open_pair_scores(flicker, close_sec=0.0, open_sec=0.30)
    self.assertEqual(int(cleaned["v2_interacting"].sum()), 0)
    self.assertGreater(stats2["n_morph_on_removed"], 0)

  def test_pipeline_morph_keeps_multi_on_zero(self):
    rows = []
    dt = 0.05
    for i in range(20):
      t = i * dt
      for obj, on in (("keyboard", 1 if 5 <= i < 12 else 0), ("mouse", 1 if 8 <= i < 10 else 0)):
        rows.append(
          {
            "timestamp_ns": i,
            "t_sec": t,
            "frame_index": i,
            "hand_side": "right",
            "object_name": obj,
            "v2_interacting": on,
            "v2_frame_ok": 1,
            "v2_co_motion_ok": on,
            "v2_stationary_ok": 0,
            "v2_aabb_dist_m": 0.04,
            "v2_dwell_ms": 300.0,
            "iou": 0.4,
          }
        )
    pair = pd.DataFrame(rows)
    clean, stats = cleanup_pair_scores(
      pair,
      TimelineCleanupConfig(
        max_gap_sec=0.12,
        min_on_sec=0.10,
        backfill_max_sec=0.0,
        morph_close_sec=0.15,
        morph_open_sec=0.30,
      ),
    )
    self.assertEqual(stats["n_multi_on_frames"], 0)

  def test_mouse_cluster_survives_default_open(self):
    dt = 0.033
    blocks = [
      (101.7, 102.6),
      (104.1, 104.9),
      (105.6, 106.0),
      (107.4, 109.1),
    ]
    t_end = 110.0
    t = []
    cur = 0.0
    while cur <= t_end:
      t.append(cur)
      cur += dt
    on = []
    for ti in t:
      flag = 0
      for a, b in blocks:
        if a <= ti < b:
          flag = 1
          break
      on.append(flag)
    pair = _lane_pair_df("right", "mouse", t, on)
    morphed, _ = morph_close_open_pair_scores(pair, close_sec=0.15, open_sec=0.30)
    segs = pred_segments_from_pair_scores(morphed)
    self.assertGreaterEqual(len(segs), 3)
    starts = [s.t_start for s in segs]
    self.assertTrue(any(abs(s - 101.7) < 0.1 for s in starts))


if __name__ == "__main__":
  unittest.main()
