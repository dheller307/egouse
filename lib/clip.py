"""Map ground-truth segment bounds to VRS clip indices (start_frame, num_frames)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from lib.eval import device_to_timecode, load_segments, load_time_mapping


@dataclass(frozen=True)
class ClipWindow:
  start_frame: int
  num_frames: int
  start_timecode_ns: int
  end_timecode_ns: int
  start_devicetime_ns: int
  end_devicetime_ns: int
  end_frame: int
  stride: int
  duration_sec: float


def segments_path_for_sequence(sequence_path: Path, egouse_root: Path) -> Path:
  """Default GT CSV: src/data/ground_truth/<sequence_id>_segments.csv."""
  return egouse_root / "src" / "data" / "ground_truth" / f"{sequence_path.name}_segments.csv"


def mapping_path_for_sequence(sequence_path: Path) -> Path:
  return sequence_path / "timecode_devicetime_mapping.csv"


def gt_devicetime_bounds(segments_path: Path) -> tuple[int, int]:
  segments_df = load_segments(segments_path)
  return int(segments_df["start_devicetime_ns"].min()), int(segments_df["end_devicetime_ns"].max())


def clip_window_from_gt(
  segments_path: Path,
  mapping_path: Path,
  timestamps: Sequence[int],
  *,
  stride: int = 1,
  pad_sec: float = 0.0,
) -> ClipWindow:
  """
  Derive start_frame and num_frames from the first/last devicetime in GT segments.

  Converts GT bounds to timecode, then finds the inclusive VRS index range in
  ``timestamps`` (TIME_CODE domain from get_sequence_timestamps).
  """
  if stride < 1:
    raise ValueError(f"stride must be >= 1, got {stride}")

  ts = np.asarray(timestamps, dtype=np.int64)
  if ts.size == 0:
    raise ValueError("timestamps is empty")

  start_devicetime_ns, end_devicetime_ns = gt_devicetime_bounds(segments_path)
  mapping_df = load_time_mapping(mapping_path)
  start_timecode_ns = device_to_timecode(start_devicetime_ns, mapping_df)
  end_timecode_ns = device_to_timecode(end_devicetime_ns, mapping_df)

  if pad_sec:
    pad_ns = int(pad_sec * 1e9)
    start_timecode_ns = max(int(ts[0]), start_timecode_ns - pad_ns)
    end_timecode_ns = min(int(ts[-1]), end_timecode_ns + pad_ns)

  if end_timecode_ns < start_timecode_ns:
    raise ValueError(
      f"GT end timecode {end_timecode_ns} is before start {start_timecode_ns}"
    )

  start_frame = int(np.searchsorted(ts, start_timecode_ns, side="left"))
  end_frame = int(np.searchsorted(ts, end_timecode_ns, side="right") - 1)

  start_frame = min(start_frame, len(ts) - 1)
  end_frame = max(end_frame, start_frame)
  end_frame = min(end_frame, len(ts) - 1)

  if end_frame < start_frame:
    raise ValueError(
      f"No VRS frames in GT window (start_frame={start_frame}, end_frame={end_frame})"
    )

  num_frames = (end_frame - start_frame) // stride + 1
  actual_end_frame = start_frame + (num_frames - 1) * stride
  duration_sec = (int(ts[actual_end_frame]) - int(ts[start_frame])) / 1e9

  return ClipWindow(
    start_frame=start_frame,
    num_frames=num_frames,
    start_timecode_ns=int(ts[start_frame]),
    end_timecode_ns=int(ts[actual_end_frame]),
    start_devicetime_ns=start_devicetime_ns,
    end_devicetime_ns=end_devicetime_ns,
    end_frame=actual_end_frame,
    stride=stride,
    duration_sec=duration_sec,
  )


def clip_window_from_full_vrs(
  timestamps: Sequence[int],
  *,
  stride: int = 1,
) -> ClipWindow:
  """Use the entire VRS timestamp range (no human GT bounds)."""
  if stride < 1:
    raise ValueError(f"stride must be >= 1, got {stride}")

  ts = np.asarray(timestamps, dtype=np.int64)
  if ts.size == 0:
    raise ValueError("timestamps is empty")

  start_frame = 0
  end_frame = len(ts) - 1
  num_frames = (end_frame - start_frame) // stride + 1
  actual_end_frame = start_frame + (num_frames - 1) * stride
  duration_sec = (int(ts[actual_end_frame]) - int(ts[start_frame])) / 1e9

  return ClipWindow(
    start_frame=start_frame,
    num_frames=num_frames,
    start_timecode_ns=int(ts[start_frame]),
    end_timecode_ns=int(ts[actual_end_frame]),
    start_devicetime_ns=0,
    end_devicetime_ns=0,
    end_frame=actual_end_frame,
    stride=stride,
    duration_sec=duration_sec,
  )


def resolve_clip_window(
  timestamps: Sequence[int],
  *,
  start_frame: int | None,
  num_frames: int | None,
  stride: int,
  segments_path: Path,
  mapping_path: Path,
  pad_sec: float = 0.0,
) -> ClipWindow:
  """
  Use explicit start_frame/num_frames when both are set; otherwise derive from GT
  when ``segments_path`` exists, or the full VRS range when it does not.

  Raises if only one of start_frame or num_frames is provided.
  """
  if start_frame is not None or num_frames is not None:
    if start_frame is None or num_frames is None:
      raise ValueError("Provide both start_frame and num_frames, or omit both to use GT bounds")
    ts = np.asarray(timestamps, dtype=np.int64)
    end_frame = start_frame + (num_frames - 1) * stride
    if start_frame < 0 or end_frame >= len(ts):
      raise ValueError(
        f"Clip window [{start_frame}, {end_frame}] (stride={stride}) "
        f"is outside timestamps length {len(ts)}"
      )
    return ClipWindow(
      start_frame=start_frame,
      num_frames=num_frames,
      start_timecode_ns=int(ts[start_frame]),
      end_timecode_ns=int(ts[end_frame]),
      start_devicetime_ns=0,
      end_devicetime_ns=0,
      end_frame=end_frame,
      stride=stride,
      duration_sec=(int(ts[end_frame]) - int(ts[start_frame])) / 1e9,
    )

  if not segments_path.is_file():
    return clip_window_from_full_vrs(timestamps, stride=stride)

  return clip_window_from_gt(
    segments_path,
    mapping_path,
    timestamps,
    stride=stride,
    pad_sec=pad_sec,
  )


def format_clip_window(window: ClipWindow) -> str:
  lines = [
    f"start_frame={window.start_frame}",
    f"num_frames={window.num_frames}",
    f"stride={window.stride}",
    f"end_frame={window.end_frame}",
    f"duration_sec={window.duration_sec:.3f}",
    f"start_timecode_ns={window.start_timecode_ns}",
    f"end_timecode_ns={window.end_timecode_ns}",
  ]
  if window.start_devicetime_ns or window.end_devicetime_ns:
    lines.append(f"start_devicetime_ns={window.start_devicetime_ns}")
    lines.append(f"end_devicetime_ns={window.end_devicetime_ns}")
  return "\n".join(lines)
