#!/usr/bin/env python3
"""Thin HOT3D Aria downloader for EgoUse teacher path.

Reads Hot3DAria_download_urls.json, downloads selected artifact groups per sequence,
verifies sha1, extracts zips into hot3d_data/<SEQ>/.

Hard-stops on HTTP 401/403/404 or checksum mismatch (refresh download_links first).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from zipfile import ZipFile, is_zipfile

EGOUSE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URLS = EGOUSE_ROOT / "download_links" / "Hot3DAria_download_urls.json"
DEFAULT_OUT = EGOUSE_ROOT / "hot3d_data"
ALLOWED_GROUPS = ("ground_truth", "hand_data", "main_vrs")
CHUNK = 1024 * 1024


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Download HOT3D Aria sequences for EgoUse.")
  p.add_argument("--seq-list", type=Path, required=True, help="One sequence ID per line")
  p.add_argument(
    "--groups",
    nargs="+",
    choices=ALLOWED_GROUPS,
    required=True,
    help="Artifact groups to download",
  )
  p.add_argument("--urls-json", type=Path, default=DEFAULT_URLS)
  p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
  p.add_argument(
    "--overwrite",
    action="store_true",
    help="Re-download even if .download_status.json marks group complete",
  )
  return p.parse_args()


def load_seq_list(path: Path) -> list[str]:
  lines = []
  for line in path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
      lines.append(line)
  if not lines:
    raise SystemExit(f"Empty seq list: {path}")
  return lines


def sha1_file(path: Path) -> str:
  h = hashlib.sha1()
  with path.open("rb") as f:
    while chunk := f.read(CHUNK):
      h.update(chunk)
  return h.hexdigest()


def download_url(url: str, dest: Path) -> int:
  req = urllib.request.Request(url, method="GET")
  try:
    with urllib.request.urlopen(req, timeout=60) as resp:
      status = resp.status
      if status != 200:
        return status
      dest.parent.mkdir(parents=True, exist_ok=True)
      with dest.open("wb") as out:
        while True:
          chunk = resp.read(CHUNK)
          if not chunk:
            break
          out.write(chunk)
      return status
  except urllib.error.HTTPError as exc:
    return exc.code


def fail_urls_expired(seq: str, group: str, status: int) -> None:
  print(
    f"\nERROR: HTTP {status} downloading {seq} / {group}.\n"
    "HOT3D CDN URLs are ephemeral (~14 days). Refresh download_links:\n"
    "  1. https://www.projectaria.com/datasets/hot3D/ → Access the Datasets\n"
    "  2. Download fresh Hot3DAria_download_urls.json\n"
    "  3. cp ~/Downloads/Hot3DAria_download_urls.json egouse/download_links/\n"
    "Then re-run this command.\n",
    file=sys.stderr,
  )
  raise SystemExit(1)


def load_status(seq_dir: Path) -> dict:
  path = seq_dir / ".download_status.json"
  if path.is_file():
    return json.loads(path.read_text())
  return {}


def save_status(seq_dir: Path, status: dict) -> None:
  path = seq_dir / ".download_status.json"
  path.write_text(json.dumps(status, indent=4) + "\n")


def group_complete(seq_dir: Path, group: str, status: dict) -> bool:
  if not status.get(group):
    return False
  if group == "main_vrs":
    return (seq_dir / "recording.vrs").is_file()
  if group == "ground_truth":
    return (seq_dir / "metadata.json").is_file() and (seq_dir / "box2d_objects.csv").is_file()
  if group == "hand_data":
    return (seq_dir / "umetrack_hand_pose_trajectory.jsonl").is_file()
  return True


def install_main_vrs(download_path: Path, seq_dir: Path) -> None:
  seq_dir.mkdir(parents=True, exist_ok=True)
  dest = seq_dir / "recording.vrs"
  if is_zipfile(download_path):
    raise RuntimeError(f"Unexpected zip for main_vrs: {download_path}")
  shutil.copy2(download_path, dest)


def extract_zip(download_path: Path, seq_dir: Path) -> None:
  seq_dir.mkdir(parents=True, exist_ok=True)
  with ZipFile(download_path) as zf:
    zf.extractall(seq_dir)


def download_group(
  seq: str,
  group: str,
  artifact: dict,
  seq_dir: Path,
  *,
  overwrite: bool,
  tmp_root: Path,
) -> None:
  url = artifact["download_url"]
  filename = artifact["filename"]
  expected_sha1 = artifact["sha1sum"]
  size_mb = artifact.get("file_size_bytes", 0) / 1e6
  print(f"  {group}: {filename} ({size_mb:.1f} MB)")

  tmp_root.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
    tmp_path = Path(tmp) / filename
    status = download_url(url, tmp_path)
    if status != 200:
      fail_urls_expired(seq, group, status)

    got = sha1_file(tmp_path)
    if got != expected_sha1:
      print(
        f"ERROR: sha1 mismatch for {seq}/{group}: got {got}, expected {expected_sha1}",
        file=sys.stderr,
      )
      raise SystemExit(1)

    if group == "main_vrs":
      install_main_vrs(tmp_path, seq_dir)
    else:
      extract_zip(tmp_path, seq_dir)

  print(f"  OK {group}")


def main() -> None:
  args = parse_args()
  if not args.urls_json.is_file():
    raise SystemExit(f"Missing URLs JSON: {args.urls_json}")

  catalog = json.loads(args.urls_json.read_text())
  sequences = catalog["sequences"]
  seq_ids = load_seq_list(args.seq_list)
  args.out_root.mkdir(parents=True, exist_ok=True)
  tmp_root = args.out_root / ".download_tmp"

  print(f"Downloading {len(seq_ids)} sequence(s) groups={args.groups}")
  print(f"URLs: {args.urls_json}")
  print(f"Out:  {args.out_root}")

  for seq in seq_ids:
    if seq not in sequences:
      raise SystemExit(f"Unknown sequence in URL JSON: {seq}")
    seq_dir = args.out_root / seq
    status = load_status(seq_dir)
    print(f"\n=== {seq} ===")
    for group in args.groups:
      if not args.overwrite and group_complete(seq_dir, group, status):
        print(f"  skip {group} (already complete)")
        status[group] = True
        continue
      if group not in sequences[seq]:
        raise SystemExit(f"Group {group} not in URL JSON for {seq}")
      download_group(
        seq, group, sequences[seq][group], seq_dir, overwrite=args.overwrite, tmp_root=tmp_root
      )
      status[group] = True
      save_status(seq_dir, status)

  print("\nDone.")


if __name__ == "__main__":
  main()
