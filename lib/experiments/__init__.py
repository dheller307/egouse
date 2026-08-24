"""Experiment runners."""

from __future__ import annotations

__all__ = ["run_gt_dual_dwell"]


def __getattr__(name: str):
  if name == "run_gt_dual_dwell":
    from lib.experiments.run_gt import run_gt_dual_dwell

    return run_gt_dual_dwell
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
