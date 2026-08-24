"""L∪R merge must apply to both pred and teacher before frame + segment metrics."""

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from lib.pipelines.image_2d_infer import eval_vs_teacher
from lib.viz.timeline import merge_hands_pair_df


def _pair_rows(*, hand: str, obj: str, on: list[int], t0: float = 0.0) -> pd.DataFrame:
  n = len(on)
  return pd.DataFrame(
    {
      "timestamp_ns": [int(1e9 * (t0 + i * 0.1)) for i in range(n)],
      "t_sec": [t0 + i * 0.1 for i in range(n)],
      "hand_side": [hand] * n,
      "object_name": [obj] * n,
      "v2_interacting": on,
    }
  )


class TestMergeHandsPairDf(unittest.TestCase):
  def test_or_across_hands(self):
    left = _pair_rows(hand="left", obj="mug", on=[1, 0, 0])
    right = _pair_rows(hand="right", obj="mug", on=[0, 1, 0])
    merged = merge_hands_pair_df(pd.concat([left, right], ignore_index=True))
    self.assertEqual(list(merged["v2_interacting"]), [1, 1, 0])
    self.assertTrue((merged["hand_side"] == "merged").all())


class TestEvalMergeHands(unittest.TestCase):
  def test_cross_hand_match_only_after_merge(self):
    # Teacher ON on left only; student ON on right only — same object/time.
    teacher = pd.concat(
      [
        _pair_rows(hand="left", obj="keyboard", on=[0, 1, 1, 0]),
        _pair_rows(hand="right", obj="keyboard", on=[0, 0, 0, 0]),
      ],
      ignore_index=True,
    )
    student = pd.concat(
      [
        _pair_rows(hand="left", obj="keyboard", on=[0, 0, 0, 0]),
        _pair_rows(hand="right", obj="keyboard", on=[0, 1, 1, 0]),
      ],
      ignore_index=True,
    )
    per_hand = eval_vs_teacher(student, teacher, iou_threshold=0.5, merge_hands=False)
    merged = eval_vs_teacher(student, teacher, iou_threshold=0.5, merge_hands=True)
    self.assertEqual(per_hand["frame_f1"], 0.0)
    self.assertEqual(per_hand["seg_f1"], 0.0)
    self.assertEqual(merged["frame_f1"], 1.0)
    self.assertEqual(merged["seg_f1"], 1.0)

  def test_merge_applies_to_pred_not_only_teacher(self):
    # If only teacher is merged, hand_side "merged" vs "left"/"right" never match.
    teacher = pd.concat(
      [
        _pair_rows(hand="left", obj="mug", on=[1, 1]),
        _pair_rows(hand="right", obj="mug", on=[0, 0]),
      ],
      ignore_index=True,
    )
    student = pd.concat(
      [
        _pair_rows(hand="left", obj="mug", on=[1, 1]),
        _pair_rows(hand="right", obj="mug", on=[0, 0]),
      ],
      ignore_index=True,
    )
    both = eval_vs_teacher(student, teacher, iou_threshold=0.5, merge_hands=True)
    teacher_only = eval_vs_teacher(
      student,
      merge_hands_pair_df(teacher),
      iou_threshold=0.5,
      merge_hands=False,
    )
    self.assertGreater(both["seg_f1"], 0.0)
    self.assertEqual(teacher_only["seg_f1"], 0.0)


if __name__ == "__main__":
  unittest.main()
