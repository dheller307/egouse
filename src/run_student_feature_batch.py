#!/usr/bin/env python3
"""Batch raw hybrid student features over train/prove sequences."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.paths import (
  EGOUSE_ROOT,
  HOT3D_DATA_DIR,
  VENV_DOH,
  VENV_MAIN,
  VENV_YOLOE,
  doh_torch_lib,
  venv_python,
)

DEFAULT_OUTPUT_ROOT = EGOUSE_ROOT / "outputs" / "student" / "features_raw"
DEFAULT_TEACHER_ROOT = EGOUSE_ROOT / "outputs" / "teacher"
MANIFEST_FIELDS = ("seq", "ok", "n_rows", "n_positive", "n_frames", "error")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Batch raw hybrid features joined to teacher y*_clean")
  p.add_argument("--sequences", nargs="*", default=[], help="Sequence names under hot3d_data/")
  p.add_argument("--seq-list", type=Path, default=None)
  p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
  p.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
  p.add_argument("--stride", type=int, default=1)
  p.add_argument("--yoloe-size", default="s")
  p.add_argument("--jpeg-quality", type=int, default=80)
  p.add_argument("--continue-on-error", action="store_true")
  p.add_argument("--skip-existing", action="store_true", help="Skip seq if features.csv exists")
  p.add_argument(
    "--tmp-root",
    type=Path,
    default=Path("/freespace/local/winlab/.tmp"),
  )
  return p.parse_args()


def load_sequence_names(args: argparse.Namespace) -> list[str]:
  names: list[str] = list(args.sequences)
  if args.seq_list is not None:
    if not args.seq_list.is_file():
      raise FileNotFoundError(args.seq_list)
    for line in args.seq_list.read_text().splitlines():
      line = line.strip()
      if line and not line.startswith("#"):
        names.append(line)
  if not names:
    raise SystemExit("Provide --sequences and/or --seq-list")
  return names


def write_manifest_row(manifest_path: Path, row: dict, *, write_header: bool) -> None:
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  mode = "w" if write_header else "a"
  with manifest_path.open(mode, newline="") as f:
    writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
    if write_header:
      writer.writeheader()
    writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def _run(cmd: list[str], *, env: dict | None = None) -> None:
  print("+", " ".join(cmd), flush=True)
  subprocess.run(cmd, check=True, env=env)


def process_sequence(seq_id: str, args: argparse.Namespace) -> dict:
  seq_path = HOT3D_DATA_DIR / seq_id
  if not seq_path.is_dir():
    raise FileNotFoundError(f"Missing sequence: {seq_path}")
  teacher_clean = args.teacher_root / seq_id / "pair_scores_clean.csv"
  if not teacher_clean.is_file():
    raise FileNotFoundError(f"Missing teacher labels: {teacher_clean}")

  out_dir = args.output_root / seq_id
  features_csv = out_dir / "features.csv"
  if args.skip_existing and features_csv.is_file():
    meta_path = out_dir / "features_meta.json"
    n_rows = n_pos = n_frames = 0
    if meta_path.is_file():
      meta = json.loads(meta_path.read_text())
      n_rows = int(meta.get("n_rows", 0))
      n_pos = int(meta.get("n_positive", 0))
      n_frames = int(meta.get("pred_frames", 0))
    return {
      "seq": seq_id,
      "ok": True,
      "n_rows": n_rows,
      "n_positive": n_pos,
      "n_frames": n_frames,
      "error": "skipped_existing",
    }

  tmp = args.tmp_root / f"student_feat_{seq_id}_{os.getpid()}"
  if tmp.exists():
    shutil.rmtree(tmp)
  tmp.mkdir(parents=True, exist_ok=True)

  venv = venv_python(VENV_MAIN)
  venv_doh = venv_python(VENV_DOH)
  venv_yoloe = venv_python(VENV_YOLOE)

  try:
    env_clean = os.environ.copy()
    env_clean.pop("LD_LIBRARY_PATH", None)

    _run(
      [
        str(venv),
        str(EGOUSE_ROOT / "scripts" / "dump_gt_boxes_for_eval.py"),
        "--sequence",
        seq_id,
        "--stride",
        str(args.stride),
        "--output-dir",
        str(tmp),
        "--jpeg-quality",
        str(args.jpeg_quality),
      ],
      env=env_clean,
    )

    env_hand = env_clean.copy()
    torch_lib = doh_torch_lib()
    if torch_lib is not None:
      env_hand["LD_LIBRARY_PATH"] = str(torch_lib)
    _run(
      [str(venv_doh), str(EGOUSE_ROOT / "src" / "extract_hybrid_hands.py"), "--eval-dir", str(tmp)],
      env=env_hand,
    )

    _run(
      [
        str(venv_yoloe),
        str(EGOUSE_ROOT / "src" / "eval_hybrid_boxes.py"),
        "--eval-dir",
        str(tmp),
        "--yoloe-size",
        args.yoloe_size,
        "--no-kalman",
      ],
      env=env_clean,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    features_out = out_dir / "features.csv"
    _run(
      [
        str(venv),
        str(EGOUSE_ROOT / "src" / "build_student_features.py"),
        "--eval-dir",
        str(tmp),
        "--teacher-clean",
        str(teacher_clean),
        "--out-csv",
        str(features_out),
      ],
      env=env_clean,
    )

    for name in ("preds.jsonl", "summary.json", "hands_preds.jsonl", "hands_meta.json", "per_object.csv"):
      src = tmp / name
      if src.is_file():
        shutil.copy2(src, out_dir / name)

    # features_meta written next to features.csv by build_student_features
    meta_src = features_out.with_name("features_meta.json")
    if not meta_src.is_file():
      # build writes <stem>_meta.json
      alt = out_dir / "features_meta.json"
      built = features_out.with_name(features_out.stem + "_meta.json")
      if built.is_file() and built != alt:
        shutil.move(str(built), str(alt))
      meta_src = alt

    meta = json.loads(meta_src.read_text()) if meta_src.is_file() else {}
    return {
      "seq": seq_id,
      "ok": True,
      "n_rows": int(meta.get("n_rows", 0)),
      "n_positive": int(meta.get("n_positive", 0)),
      "n_frames": int(meta.get("pred_frames", 0)),
      "error": "",
    }
  finally:
    if tmp.exists():
      shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
  args = parse_args()
  args.tmp_root.mkdir(parents=True, exist_ok=True)
  names = load_sequence_names(args)
  args.output_root.mkdir(parents=True, exist_ok=True)
  manifest_path = args.output_root / "manifest.csv"

  avail = shutil.disk_usage("/freespace").free
  if avail < 4 * 1024**3:
    raise SystemExit(f"Need >=4 GiB free on /freespace (have {avail})")

  print(f"Batch student features: {len(names)} seq(s) → {args.output_root}", flush=True)
  n_ok = n_fail = 0
  write_header = not manifest_path.is_file()
  for i, seq_id in enumerate(names):
    print(f"\n--- [{i + 1}/{len(names)}] {seq_id} ---", flush=True)
    row = {"seq": seq_id, "ok": False, "n_rows": 0, "n_positive": 0, "n_frames": 0, "error": ""}
    try:
      row = process_sequence(seq_id, args)
      print(
        f"OK: rows={row['n_rows']} pos={row['n_positive']} frames={row['n_frames']}"
        + (f" ({row['error']})" if row.get("error") == "skipped_existing" else ""),
        flush=True,
      )
      n_ok += 1
    except Exception as exc:
      row["error"] = str(exc)
      print(f"FAIL: {exc}", flush=True)
      n_fail += 1
      write_manifest_row(manifest_path, row, write_header=write_header)
      write_header = False
      if not args.continue_on_error:
        raise
      continue
    write_manifest_row(manifest_path, row, write_header=write_header)
    write_header = False

  print(f"\nManifest: {manifest_path}", flush=True)
  print(f"Done: {n_ok} ok, {n_fail} failed", flush=True)
  if n_fail:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
