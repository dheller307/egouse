"""Lightweight signal processing for timeline confidence curves."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
  if window <= 1:
    return np.asarray(y, dtype=float)
  return pd.Series(y).rolling(window, min_periods=1).mean().to_numpy()


def threshold_binary(y: np.ndarray, threshold: float) -> np.ndarray:
  return (np.asarray(y, dtype=float) >= threshold).astype(float)


def apply_gap_fill(binary: np.ndarray, t_sec: np.ndarray, max_gap_sec: float) -> np.ndarray:
  """Bridge short OFF gaps between ON runs (forgive brief dips below threshold)."""
  out = np.asarray(binary, dtype=bool).copy()
  t_sec = np.asarray(t_sec, dtype=float)
  n = len(out)
  if n == 0 or max_gap_sec <= 0:
    return out.astype(float)

  i = 0
  while i < n:
    while i < n and not out[i]:
      i += 1
    if i >= n:
      break
    while i < n and out[i]:
      i += 1
    gap_start = i
    j = gap_start
    while j < n and not out[j]:
      j += 1
    if j > gap_start and j < n:
      gap_duration = t_sec[j - 1] - t_sec[gap_start]
      if gap_duration < max_gap_sec:
        out[gap_start:j] = True
    i = j
  return out.astype(float)


def apply_dwell(binary: np.ndarray, t_sec: np.ndarray, min_dwell_sec: float) -> np.ndarray:
  """Zero out short runs of 1s shorter than min_dwell_sec."""
  out = np.asarray(binary, dtype=bool).copy()
  t_sec = np.asarray(t_sec, dtype=float)
  n = len(out)
  if n == 0 or min_dwell_sec <= 0:
    return out.astype(float)

  i = 0
  while i < n:
    if not out[i]:
      i += 1
      continue
    j = i + 1
    while j < n and out[j]:
      j += 1
    duration = t_sec[j - 1] - t_sec[i]
    if duration < min_dwell_sec:
      out[i:j] = False
    i = j
  return out.astype(float)


def process_confidence(
  t_sec: np.ndarray,
  y: np.ndarray,
  *,
  smooth_window: int = 1,
  threshold: float | None = None,
  gap_sec: float = 0.0,
  dwell_sec: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None]:
  """
  Smooth a confidence trace and optionally derive a binary processed series.

  Pipeline: smooth → threshold → gap_fill → dwell_on

  Returns (y_smooth, y_binary). y_binary is None when threshold is None.
  """
  y_smooth = rolling_mean(np.asarray(y, dtype=float), smooth_window)
  if threshold is None:
    return y_smooth, None
  y_binary = threshold_binary(y_smooth, threshold)
  if gap_sec > 0:
    y_binary = apply_gap_fill(y_binary, t_sec, gap_sec)
  if dwell_sec > 0:
    y_binary = apply_dwell(y_binary, t_sec, dwell_sec)
  return y_smooth, y_binary
