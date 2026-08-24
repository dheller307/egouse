"""Unit tests for timeline segment metrics."""

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from lib.eval_timeline import (
  TimelineSegment,
  edit_score,
  evaluate_timeline,
  gt_segments_from_timeline_df,
  on_intervals,
  pred_segments_from_pair_scores,
  segment_f1_at_iou,
  temporal_iou,
)


class TestOnIntervals(unittest.TestCase):
  def test_merge_runs(self):
    t = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    on = np.array([False, True, True, False, True])
    iv = on_intervals(t, on)
    self.assertEqual(len(iv), 2)
    self.assertAlmostEqual(iv[0][0], 0.1)
    self.assertAlmostEqual(iv[0][1], 0.3)
    self.assertAlmostEqual(iv[1][0], 0.4)

  def test_sparse_hole_does_not_span(self):
    # Dense ~0.033s samples then a 1.4s hole — must not paint through the hole.
    t = np.array([0.0, 0.033, 0.066, 1.5, 1.533])
    on = np.array([True, True, True, False, False])
    iv = on_intervals(t, on)
    self.assertEqual(len(iv), 1)
    self.assertAlmostEqual(iv[0][0], 0.0)
    self.assertLess(iv[0][1], 0.2)


class TestSegmentF1(unittest.TestCase):
  def test_perfect_match(self):
    gt = [TimelineSegment("left", "mug_white", 1.0, 3.0)]
    pred = [TimelineSegment("left", "mug_white", 1.0, 3.0)]
    m = segment_f1_at_iou(gt, pred, iou_threshold=0.5)
    self.assertEqual(m["seg_f1"], 1.0)
    self.assertEqual(m["n_matched"], 1)

  def test_wrong_object_no_match(self):
    gt = [TimelineSegment("left", "mug_white", 1.0, 3.0)]
    pred = [TimelineSegment("left", "keyboard", 1.0, 3.0)]
    m = segment_f1_at_iou(gt, pred, iou_threshold=0.5)
    self.assertEqual(m["n_matched"], 0)
    self.assertEqual(m["seg_f1"], 0.0)

  def test_partial_overlap_below_threshold(self):
    gt = [TimelineSegment("right", "mouse", 0.0, 10.0)]
    pred = [TimelineSegment("right", "mouse", 0.0, 2.0)]  # IoU = 2/10 = 0.2
    self.assertLess(temporal_iou(gt[0], pred[0]), 0.5)
    m = segment_f1_at_iou(gt, pred, iou_threshold=0.5)
    self.assertEqual(m["n_matched"], 0)

  def test_greedy_one_to_one(self):
    gt = [
      TimelineSegment("left", "a", 0.0, 2.0),
      TimelineSegment("left", "a", 4.0, 6.0),
    ]
    pred = [TimelineSegment("left", "a", 0.0, 2.0)]
    m = segment_f1_at_iou(gt, pred, iou_threshold=0.5)
    self.assertEqual(m["n_matched"], 1)
    self.assertEqual(m["seg_rec"], 0.5)
    self.assertEqual(m["seg_prec"], 1.0)


class TestEdit(unittest.TestCase):
  def test_identical_sequences(self):
    segs = [
      TimelineSegment("left", "keyboard", 0.0, 1.0),
      TimelineSegment("left", "mug_white", 2.0, 3.0),
    ]
    m = edit_score(segs, list(segs))
    self.assertEqual(m["edit_distance"], 0)
    self.assertEqual(m["edit_sim"], 1.0)

  def test_insertion_lowers_sim(self):
    gt = [TimelineSegment("left", "keyboard", 0.0, 1.0)]
    pred = [
      TimelineSegment("left", "keyboard", 0.0, 1.0),
      TimelineSegment("left", "mouse", 1.0, 2.0),
    ]
    m = edit_score(gt, pred)
    self.assertEqual(m["edit_distance"], 1)
    self.assertEqual(m["edit_sim"], 0.5)


class TestFromDataframes(unittest.TestCase):
  def test_pred_and_gt_extraction(self):
    pair = pd.DataFrame(
      {
        "hand_side": ["left"] * 4,
        "object_name": ["keyboard"] * 4,
        "t_sec": [0.0, 0.1, 0.2, 0.3],
        "v2_interacting": [0, 1, 1, 0],
      }
    )
    pred = pred_segments_from_pair_scores(pair)
    self.assertEqual(len(pred), 1)
    self.assertEqual(pred[0].object_name, "keyboard")

    timeline = pd.DataFrame(
      {
        "hand_side": ["left"] * 4,
        "t_sec": [0.0, 0.1, 0.2, 0.3],
        "gt_object_name": ["none", "keyboard", "keyboard", "none"],
        "gt_interacting": [False, True, True, False],
      }
    )
    gt = gt_segments_from_timeline_df(timeline)
    self.assertEqual(len(gt), 1)
    m = evaluate_timeline(gt, pred, iou_threshold=0.5)
    self.assertEqual(m["seg_f1"], 1.0)
    self.assertEqual(m["edit_sim"], 1.0)


if __name__ == "__main__":
  unittest.main()
