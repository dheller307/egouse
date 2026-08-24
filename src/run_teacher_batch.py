#!/usr/bin/env python3
"""Batch frozen teacher labels over multiple HOT3D sequences."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.paths import EGOUSE_ROOT, HOT3D_DATA_DIR, setup_paths

setup_paths()

from lib.experiments.teacher_pipeline import FROZEN_CLEANUP_CONFIG, run_teacher_pipeline
from lib.scoring.dual_dwell_config import FROZEN_DUAL_DWELL

DEFAULT_OUTPUT_ROOT = EGOUSE_ROOT / "outputs" / "teacher"
MANIFEST_FIELDS = ("seq", "ok", "n_pair_rows", "n_clean_on", "has_gt", "error")


def parse_args():
  p = argparse.ArgumentParser(
    description="Batch frozen gt_world teacher: score → cleanup → y*_clean per sequence."
  )
  p.add_argument(
    "--sequences",
    nargs="*",
    default=[],
    help="Sequence names (under hot3d_data/) or absolute paths",
  )
  p.add_argument(
    "--seq-list",
    type=Path,
    default=None,
    help="Text file with one sequence name or path per line",
  )
  p.add_argument(
    "--output-root",
    type=Path,
    default=DEFAULT_OUTPUT_ROOT,
    help="Root directory; each seq writes to <output-root>/<SEQ>/",
  )
  p.add_argument(
    "--object-library-path",
    type=Path,
    default=HOT3D_DATA_DIR / "assets",
  )
  p.add_argument(
    "--skip-score",
    action="store_true",
    help="Reuse existing pair_scores.csv in each seq output dir",
  )
  p.add_argument(
    "--continue-on-error",
    action="store_true",
    help="Keep processing remaining sequences after a failure",
  )
  return p.parse_args()


def resolve_sequence_path(name_or_path: str, hot3d_data_dir: Path) -> Path:
  p = Path(name_or_path)
  if p.is_absolute() and p.is_dir():
    return p
  candidate = hot3d_data_dir / name_or_path
  if candidate.is_dir():
    return candidate
  raise FileNotFoundError(f"Sequence not found: {name_or_path}")


def load_sequence_list(args) -> list[Path]:
  names: list[str] = list(args.sequences)
  if args.seq_list is not None:
    if not args.seq_list.is_file():
      raise FileNotFoundError(f"Missing seq list: {args.seq_list}")
    for line in args.seq_list.read_text().splitlines():
      line = line.strip()
      if line and not line.startswith("#"):
        names.append(line)
  if not names:
    raise SystemExit("Provide --sequences and/or --seq-list")
  return [resolve_sequence_path(n, HOT3D_DATA_DIR) for n in names]


def write_manifest_row(manifest_path: Path, row: dict, *, write_header: bool) -> None:
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  mode = "w" if write_header else "a"
  with manifest_path.open(mode, newline="") as f:
    writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
    if write_header:
      writer.writeheader()
    writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def main():
  args = parse_args()
  sequences = load_sequence_list(args)
  output_root = Path(args.output_root)
  output_root.mkdir(parents=True, exist_ok=True)
  manifest_path = output_root / "manifest.csv"

  print(f"Batch teacher: {len(sequences)} sequence(s) → {output_root}")
  print(f"Preset: {FROZEN_DUAL_DWELL.name} (frozen)")

  n_ok = 0
  n_fail = 0
  for i, seq_path in enumerate(sequences):
    seq_id = seq_path.name
    out_dir = output_root / seq_id
    print(f"\n--- [{i + 1}/{len(sequences)}] {seq_id} ---")
    row = {
      "seq": seq_id,
      "ok": False,
      "n_pair_rows": 0,
      "n_clean_on": 0,
      "has_gt": False,
      "error": "",
    }
    try:
      result = run_teacher_pipeline(
        seq_path,
        args.object_library_path,
        out_dir,
        preset=FROZEN_DUAL_DWELL,
        cleanup_cfg=FROZEN_CLEANUP_CONFIG,
        skip_score=args.skip_score,
        preset_key="frozen",
      )
      row["ok"] = True
      row["n_pair_rows"] = result.n_pair_rows
      row["n_clean_on"] = result.n_clean_on
      row["has_gt"] = result.has_human_gt
      stats = result.cleanup_stats
      print(
        f"OK: {result.n_pair_rows} pair rows, {result.n_clean_on} clean ON; "
        f"multi-ON={stats['n_multi_on_frames']}"
      )
      if result.frame_clean:
        fm = result.frame_clean
        if result.timeline_clean:
          print(
            f"  strict: s_prec={fm.get('s_prec', float('nan')):.3f}  "
            f"seg_f1={result.timeline_clean.get('seg_f1', float('nan')):.3f}"
          )
        else:
          print(f"  strict: s_prec={fm.get('s_prec', float('nan')):.3f}")
      n_ok += 1
    except Exception as exc:
      row["error"] = str(exc)
      print(f"FAIL: {exc}")
      n_fail += 1
      if not args.continue_on_error:
        write_manifest_row(manifest_path, row, write_header=(i == 0))
        raise

    write_manifest_row(manifest_path, row, write_header=(i == 0))

  print(f"\nManifest: {manifest_path}")
  print(f"Done: {n_ok} ok, {n_fail} failed")
  if n_fail:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
