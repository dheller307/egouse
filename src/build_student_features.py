#!/usr/bin/env python3
"""Build raw hybrid association + temporal features joined to teacher y*_clean."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.geometry import bbox_center, bbox_iou, norm_center_distance
from lib.paths import EGOUSE_ROOT
from lib.student_features import FEATURE_COLUMNS, ROLL_K, ROLL_MIN_VALID


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Student features from raw hybrid preds + teacher y*")
  parser.add_argument(
    "--eval-dir",
    type=Path,
    default=EGOUSE_ROOT / "outputs" / "student" / "eval_hybrid_raw_p0003",
  )
  parser.add_argument(
    "--teacher-clean",
    type=Path,
    default=EGOUSE_ROOT / "outputs" / "teacher" / "P0003_c701bd11" / "pair_scores_clean.csv",
  )
  parser.add_argument(
    "--out-csv",
    type=Path,
    default=None,
    help="Default: <eval-dir>/features.csv",
  )
  parser.add_argument("--roll-k", type=int, default=ROLL_K)
  parser.add_argument("--max-rows", type=int, default=None, help="Cap rows for quick smoke")
  return parser.parse_args()


def _load_preds_by_ts(path: Path) -> dict[int, dict]:
  by_ts: dict[int, dict] = {}
  with path.open() as f:
    for line in f:
      rec = json.loads(line)
      for obj in rec.get("objects") or []:
        if obj.get("coasting"):
          raise ValueError(
            f"coasting object in preds at ts={rec.get('timestamp_ns')} — refuse Kalman preds"
          )
      by_ts[int(rec["timestamp_ns"])] = rec
  return by_ts


def _best_hand_for_side(hands: list[dict], side: str) -> dict | None:
  sided = [h for h in hands if h.get("side") == side]
  if sided:
    return max(sided, key=lambda h: float(h.get("score") or 0.0))
  return max(hands, key=lambda h: float(h.get("score") or 0.0)) if hands else None


def _best_obj_for_name(objects: list[dict], object_name: str) -> dict | None:
  named = [o for o in objects if o.get("name") == object_name]
  if named:
    return max(named, key=lambda o: float(o.get("score") or 0.0))
  return None


def _box_area(box: tuple[float, float, float, float]) -> float:
  l, t, r, b = box
  return max(0.0, r - l) * max(0.0, b - t)


def _pair_static(hand: dict | None, obj: dict | None) -> dict:
  nan = float("nan")
  out = {
    "hand_obj_iou": nan,
    "hand_obj_center_dist_norm": nan,
    "area_ratio": nan,
    "hand_score": nan,
    "hand_contact": nan,
    "hand_offset_x": nan,
    "hand_offset_y": nan,
    "obj_score": nan,
    "obj_present": 0,
    "hand_cx": nan,
    "hand_cy": nan,
    "obj_cx": nan,
    "obj_cy": nan,
  }
  if hand is not None:
    out["hand_score"] = float(hand.get("score") if hand.get("score") is not None else nan)
    contact = hand.get("contact")
    out["hand_contact"] = float(contact) if contact is not None else nan
    off = hand.get("offset")
    if off is not None and len(off) >= 2:
      out["hand_offset_x"] = float(off[0])
      out["hand_offset_y"] = float(off[1])
    hc = bbox_center(tuple(hand["box"]))
    if hc is not None:
      out["hand_cx"], out["hand_cy"] = float(hc[0]), float(hc[1])

  if obj is not None:
    out["obj_present"] = 1
    out["obj_score"] = float(obj.get("score") if obj.get("score") is not None else nan)
    oc = bbox_center(tuple(obj["box"]))
    if oc is not None:
      out["obj_cx"], out["obj_cy"] = float(oc[0]), float(oc[1])

  if hand is not None and obj is not None:
    hbox = tuple(hand["box"])
    obox = tuple(obj["box"])
    out["hand_obj_iou"] = float(bbox_iou(hbox, obox) or 0.0)
    dist = norm_center_distance(hbox, obox)
    out["hand_obj_center_dist_norm"] = float(dist) if dist is not None else nan
    ha = _box_area(hbox)
    oa = _box_area(obox)
    out["area_ratio"] = (ha / oa) if oa > 0 else nan
  return out


def _roll_stats(values: list[float], times: list[float], k: int) -> tuple[float, float, float]:
  """Return mean, std, slope over last k samples (NaN if < ROLL_MIN_VALID finite)."""
  nan = float("nan")
  window_v = values[-k:]
  window_t = times[-k:]
  pairs = [(t, v) for t, v in zip(window_t, window_v) if v == v]  # finite check
  if len(pairs) < ROLL_MIN_VALID:
    return nan, nan, nan
  vs = np.asarray([p[1] for p in pairs], dtype=float)
  ts = np.asarray([p[0] for p in pairs], dtype=float)
  mean = float(vs.mean())
  std = float(vs.std(ddof=0))
  if len(vs) >= 2 and float(ts.max() - ts.min()) > 0:
    slope = float(np.polyfit(ts, vs, 1)[0])
  else:
    slope = nan
  return mean, std, slope


def _add_temporal(df: pd.DataFrame, roll_k: int) -> pd.DataFrame:
  nan = float("nan")
  df = df.sort_values(["hand_side", "object_name", "timestamp_ns"]).reset_index(drop=True)
  out_rel = pd.Series(nan, index=df.index, dtype=float)
  out_iou_m = pd.Series(nan, index=df.index, dtype=float)
  out_dist_m = pd.Series(nan, index=df.index, dtype=float)
  out_dist_s = pd.Series(nan, index=df.index, dtype=float)
  out_dist_sl = pd.Series(nan, index=df.index, dtype=float)
  out_frac = pd.Series(nan, index=df.index, dtype=float)

  for (_, _), g in df.groupby(["hand_side", "object_name"], sort=False):
    idxs = g.index.to_list()
    prev_r = None
    prev_ts = None
    iou_hist: list[float] = []
    dist_hist: list[float] = []
    t_hist: list[float] = []
    present_hist: list[int] = []
    for idx in idxs:
      row = df.loc[idx]
      ts = float(row["timestamp_ns"])
      t_sec = float(row["t_sec"]) if row["t_sec"] == row["t_sec"] else ts / 1e9
      hx, hy = row["hand_cx"], row["hand_cy"]
      ox, oy = row["obj_cx"], row["obj_cy"]

      rel = nan
      if (
        hx == hx
        and hy == hy
        and ox == ox
        and oy == oy
        and prev_r is not None
        and prev_ts is not None
        and ts > prev_ts
      ):
        r = (float(hx) - float(ox), float(hy) - float(oy))
        dt = (ts - prev_ts) / 1e9
        if dt > 0:
          rel = math.hypot(r[0] - prev_r[0], r[1] - prev_r[1]) / dt
        prev_r = r
      elif hx == hx and hy == hy and ox == ox and oy == oy:
        prev_r = (float(hx) - float(ox), float(hy) - float(oy))
      else:
        prev_r = None
      prev_ts = ts

      iou_hist.append(float(row["hand_obj_iou"]))
      dist_hist.append(float(row["hand_obj_center_dist_norm"]))
      t_hist.append(t_sec)
      present_hist.append(int(row["obj_present"]))

      iou_m, _, _ = _roll_stats(iou_hist, t_hist, roll_k)
      dist_m, dist_s, dist_sl = _roll_stats(dist_hist, t_hist, roll_k)
      present_window = present_hist[-roll_k:]
      frac = float(sum(present_window) / len(present_window)) if present_window else nan

      out_rel.loc[idx] = rel
      out_iou_m.loc[idx] = iou_m
      out_dist_m.loc[idx] = dist_m
      out_dist_s.loc[idx] = dist_s
      out_dist_sl.loc[idx] = dist_sl
      out_frac.loc[idx] = frac

  df["rel_speed"] = out_rel
  df["iou_roll_mean"] = out_iou_m
  df["dist_roll_mean"] = out_dist_m
  df["dist_roll_std"] = out_dist_s
  df["dist_roll_slope"] = out_dist_sl
  df["obj_present_frac"] = out_frac
  return df


def main() -> None:
  args = parse_args()
  preds_path = args.eval_dir / "preds.jsonl"
  if not preds_path.is_file():
    raise FileNotFoundError(preds_path)
  if not args.teacher_clean.is_file():
    raise FileNotFoundError(args.teacher_clean)

  preds_by_ts = _load_preds_by_ts(preds_path)
  teacher = pd.read_csv(args.teacher_clean)

  rows = []
  for _, tr in teacher.iterrows():
    ts = int(tr["timestamp_ns"])
    pred = preds_by_ts.get(ts)
    if pred is None:
      continue

    hand_side = str(tr["hand_side"])
    object_name = str(tr["object_name"])
    hand = _best_hand_for_side(pred.get("hands") or [], hand_side)
    obj = _best_obj_for_name(pred.get("objects") or [], object_name)
    static = _pair_static(hand, obj)
    t_sec = float(tr["t_sec"]) if "t_sec" in tr and tr["t_sec"] == tr["t_sec"] else float("nan")

    rows.append(
      {
        "timestamp_ns": ts,
        "t_sec": t_sec,
        "hand_side": hand_side,
        "object_name": object_name,
        **static,
        "y_interacting": int(tr.get("v2_interacting") or 0),
      }
    )
    if args.max_rows and len(rows) >= args.max_rows:
      break

  out = pd.DataFrame(rows)
  if len(out):
    out = _add_temporal(out, args.roll_k)

  keep = (
    ["timestamp_ns", "t_sec", "hand_side", "object_name"]
    + FEATURE_COLUMNS
    + ["y_interacting"]
  )
  # Drop helper cols used only for temporal compute
  out = out[[c for c in keep if c in out.columns]]

  out_path = args.out_csv or (args.eval_dir / "features.csv")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out.to_csv(out_path, index=False)

  meta = {
    "backend": "raw_hybrid",
    "n_rows": len(out),
    "n_positive": int(out["y_interacting"].sum()) if len(out) else 0,
    "pred_frames": len(preds_by_ts),
    "teacher_path": str(args.teacher_clean),
    "feature_columns": FEATURE_COLUMNS,
    "roll_k": args.roll_k,
  }
  meta_path = out_path.with_name(out_path.stem + "_meta.json")
  meta_path.write_text(json.dumps(meta, indent=2))
  print(json.dumps(meta, indent=2))
  print(f"wrote {out_path}")


if __name__ == "__main__":
  main()
