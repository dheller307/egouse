#!/usr/bin/env python3
"""Aggregate P0003 + prove metrics into markdown/JSON. Public tables use L∪R merge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.paths import DEFAULT_STUDENT_CKPT, EGOUSE_ROOT
from lib.pipelines.image_2d_infer import eval_pred_vs_human_gt, eval_vs_teacher
from lib.student_sequence_data import load_seq_list

P0003 = "P0003_c701bd11"
DEFAULT_CKPT = DEFAULT_STUDENT_CKPT
TEACHER_ROOT = EGOUSE_ROOT / "outputs" / "teacher"
GRU_PROVE_MERGED = EGOUSE_ROOT / "outputs" / "student" / "eval_prove_phase_d_r7_merged"
GRU_TRAIN_MERGED = EGOUSE_ROOT / "outputs" / "student" / "eval_train_phase_d_r7_merged"
GRU_P0003_MERGED = EGOUSE_ROOT / "outputs" / "student" / "eval_p0003_phase_d_r7_merged"
IMAGE2D_PROVE_MERGED = EGOUSE_ROOT / "outputs" / "baseline" / "image_2d_det_prove_merged"
IMAGE2D_P0003_MERGED = EGOUSE_ROOT / "outputs" / "baseline" / "image_2d_det_p0003_merged"
IMAGE2D_P0003_PAIRS = EGOUSE_ROOT / "outputs" / "baseline" / "image_2d_det_p0003"
GRU_P0003_PAIRS = EGOUSE_ROOT / "outputs" / "student" / "eval_p0003_phase_d_r7"


def _load_json(path: Path) -> dict | None:
  if path.is_file():
    return json.loads(path.read_text())
  return None


def _round3(d: dict, keys: tuple[str, ...]) -> dict:
  out = {}
  for k in keys:
    v = d.get(k)
    out[k] = round(float(v), 3) if v is not None else None
  return out


def _metrics_subset(d: dict | None) -> dict:
  if not d:
    return {}
  keys = (
    "frame_prec",
    "frame_rec",
    "frame_f1",
    "seg_f1",
    "seg_prec",
    "seg_rec",
    "edit_sim",
    "n_rows",
    "n_positive_true",
    "n_positive_pred",
    "n_pred",
    "n_matched",
    "merge_hands",
  )
  out = {k: d[k] for k in keys if k in d}
  if "mean_frame_f1" in d and "frame_f1" not in out:
    out["frame_f1"] = d["mean_frame_f1"]
    out["frame_prec"] = d.get("mean_frame_prec")
    out["frame_rec"] = d.get("mean_frame_rec")
    out["seg_f1"] = d.get("mean_seg_f1")
    out["seg_prec"] = d.get("mean_seg_prec")
    out["seg_rec"] = d.get("mean_seg_rec")
    out["edit_sim"] = d.get("mean_edit_sim")
    out["merge_hands"] = d.get("merge_hands")
    out["n_seqs"] = d.get("n_seqs")
  return out


def _gru_p0003_merged() -> dict | None:
  p = GRU_P0003_MERGED / "prove_summary.json"
  s = _load_json(p)
  if s and s.get("per_seq"):
    return s["per_seq"][0]
  seq_sum = GRU_P0003_MERGED / "prove" / P0003 / "summary.json"
  return _load_json(seq_sum)


def _image2d_p0003_merged() -> dict | None:
  p = IMAGE2D_P0003_MERGED / P0003 / "summary_vs_teacher.json"
  hit = _load_json(p)
  if hit:
    return hit
  alt = IMAGE2D_P0003_MERGED / "prove_summary_vs_teacher.json"
  s = _load_json(alt)
  if s and s.get("per_seq"):
    return s["per_seq"][0]
  return None


def _ensure_p0003_vs_teacher() -> tuple[dict, dict]:
  gru = _gru_p0003_merged()
  image2d = _image2d_p0003_merged()

  if gru is None:
    import torch

    from src.eval_student_gru import eval_sequence
    from lib.student_gru_infer import load_gru_checkpoint

    device = torch.device("cpu")
    model, vocab, obj_to_id, _hparams = load_gru_checkpoint(DEFAULT_CKPT, device)
    gru = eval_sequence(
      model,
      vocab,
      obj_to_id,
      device,
      P0003,
      EGOUSE_ROOT / "outputs" / "student" / "features_raw",
      TEACHER_ROOT,
      EGOUSE_ROOT / "outputs" / "student" / "_aggregate_tmp_merged",
      merged_eval=True,
    )

  if image2d is None:
    pred_path = IMAGE2D_P0003_PAIRS / P0003 / "pair_scores_clean.csv"
    teacher = pd.read_csv(TEACHER_ROOT / P0003 / "pair_scores_clean.csv")
    if pred_path.is_file():
      pred = pd.read_csv(pred_path)
      image2d = eval_vs_teacher(pred, teacher, merge_hands=True)
      image2d["seq"] = P0003
      image2d["box_source"] = "detector_preds"
    else:
      from lib.pipelines.image_2d_infer import run_image_2d_for_sequence

      image2d = run_image_2d_for_sequence(
        P0003,
        preds_root=EGOUSE_ROOT / "outputs" / "student" / "features_raw",
        teacher_root=TEACHER_ROOT,
        output_dir=IMAGE2D_P0003_MERGED,
        merge_hands=True,
      )
  return image2d, gru


def _p0003_vs_gt(image2d_pred: pd.DataFrame | None, gru_pred: pd.DataFrame | None) -> dict:
  teacher_pred = pd.read_csv(TEACHER_ROOT / P0003 / "pair_scores_clean.csv")
  teacher_gt = eval_pred_vs_human_gt(
    teacher_pred, P0003, teacher_root=TEACHER_ROOT, merge_hands=True
  )
  out = {"teacher": teacher_gt}
  if image2d_pred is not None:
    out["image_2d"] = eval_pred_vs_human_gt(
      image2d_pred, P0003, teacher_root=TEACHER_ROOT, merge_hands=True
    )
  if gru_pred is not None:
    out["gru"] = eval_pred_vs_human_gt(
      gru_pred, P0003, teacher_root=TEACHER_ROOT, merge_hands=True
    )
  return out


def collect_metrics() -> dict:
  image2d, gru = _ensure_p0003_vs_teacher()

  image2d_pred_path = IMAGE2D_P0003_PAIRS / P0003 / "pair_scores_clean.csv"
  gru_pred_path = GRU_P0003_PAIRS / "prove" / P0003 / "pair_scores_student.csv"
  if not gru_pred_path.is_file():
    gru_pred_path = (
      EGOUSE_ROOT / "outputs" / "student" / "infer" / P0003 / "pair_scores.csv"
    )
  image2d_pred = pd.read_csv(image2d_pred_path) if image2d_pred_path.is_file() else None
  gru_pred = pd.read_csv(gru_pred_path) if gru_pred_path.is_file() else None
  vs_gt = _p0003_vs_gt(image2d_pred, gru_pred)

  prove_gru = _load_json(GRU_PROVE_MERGED / "prove_summary.json")
  prove_img = _load_json(IMAGE2D_PROVE_MERGED / "prove_summary_vs_teacher.json")
  train_gru = _load_json(GRU_TRAIN_MERGED / "prove_summary.json")

  teacher_tl = _load_json(TEACHER_ROOT / P0003 / "summary_teacher_timeline.json")
  teacher_frame_occ = (teacher_tl or {}).get("frame_clean") or {}
  teacher_seg_occ = (teacher_tl or {}).get("timeline_clean") or {}

  return {
    "eval_mode": "merged_LunionR",
    "eval_note": (
      "Public metrics are pair-level after L∪R merge: ON if either hand is ON "
      "per (timestamp_ns, object_name). Same rule as merge_hands_pair_df / swimlane plots."
    ),
    "seq": P0003,
    "image_2d_vs_teacher": image2d,
    "gru_vs_teacher": gru,
    "vs_human_gt_merged_pair": vs_gt,
    "teacher_vs_gt_frame_occupancy_per_hand": teacher_frame_occ,
    "teacher_vs_gt_timeline_per_hand": teacher_seg_occ,
    "prove_vs_teacher": {
      "image_2d": _metrics_subset(prove_img),
      "gru_phase_d_r7": _metrics_subset(prove_gru),
    },
    "train_vs_teacher": {
      "gru_phase_d_r7": _metrics_subset(train_gru),
    },
  }


def render_markdown(data: dict) -> str:
  img = data["image_2d_vs_teacher"]
  gru = data["gru_vs_teacher"]
  gt = data.get("vs_human_gt_merged_pair") or {}
  prove = data.get("prove_vs_teacher") or {}
  train = (data.get("train_vs_teacher") or {}).get("gru_phase_d_r7") or {}

  def _f(d: dict, k: str) -> str:
    v = d.get(k)
    return f"{float(v):.3f}" if v is not None else "—"

  img_gt = gt.get("image_2d") or {}
  gru_gt = gt.get("gru") or {}
  tch_gt = gt.get("teacher") or {}
  prove_img = prove.get("image_2d") or {}
  prove_gru = prove.get("gru_phase_d_r7") or {}

  lines = [
    f"# Results — L∪R merged (auto-generated)",
    "",
    "**Eval standard:** pair-level after `merge_hands_pair_df` (ON if either hand is ON).",
    "**Student:** `checkpoints/phase_d_r7/best.pt` (phase_d r7, trained per-hand).",
    "**image_2d:** detector preds (100DOH + YOLOE).",
    "",
    f"## `{P0003}` vs teacher `y*_clean`",
    "",
    "| Method | Frame prec | Frame F1 | Event seg_prec | Event seg_f1 |",
    "|--------|------------|----------|----------------|--------------|",
    f"| **image_2d** (detector) | {_f(img, 'frame_prec')} | {_f(img, 'frame_f1')} | {_f(img, 'seg_prec')} | {_f(img, 'seg_f1')} |",
    f"| **GRU student** | {_f(gru, 'frame_prec')} | {_f(gru, 'frame_f1')} | {_f(gru, 'seg_prec')} | {_f(gru, 'seg_f1')} |",
    "",
    "## Same clip vs human GT (merged pair-level)",
    "",
    "| Method | Frame prec | Frame F1 | Event seg_prec | Event seg_f1 |",
    "|--------|------------|----------|----------------|--------------|",
    f"| image_2d (detector) | {_f(img_gt, 'frame_prec')} | {_f(img_gt, 'frame_f1')} | {_f(img_gt, 'seg_prec')} | {_f(img_gt, 'seg_f1')} |",
    f"| GRU phase_d r7 | {_f(gru_gt, 'frame_prec')} | {_f(gru_gt, 'frame_f1')} | {_f(gru_gt, 'seg_prec')} | {_f(gru_gt, 'seg_f1')} |",
    f"| Teacher `y_clean` (ref) | {_f(tch_gt, 'frame_prec')} | {_f(tch_gt, 'frame_f1')} | {_f(tch_gt, 'seg_prec')} | {_f(tch_gt, 'seg_f1')} |",
    "",
    "## Prove mean (n=10) vs teacher",
    "",
    "| Method | Frame F1 | Event seg_f1 |",
    "|--------|----------|--------------|",
    f"| image_2d (detector) | {_f(prove_img, 'frame_f1')} | {_f(prove_img, 'seg_f1')} |",
    f"| GRU phase_d r7 | {_f(prove_gru, 'frame_f1')} | {_f(prove_gru, 'seg_f1')} |",
    "",
  ]
  if train:
    lines.extend(
      [
        "## Train mean (n=122) vs teacher (GRU)",
        "",
        f"| Frame F1 | Event seg_f1 |",
        f"|----------|--------------|",
        f"| {_f(train, 'frame_f1')} | {_f(train, 'seg_f1')} |",
        "",
      ]
    )
  return "\n".join(lines)


def write_prove_json(data: dict, path: Path) -> None:
  prove = data.get("prove_vs_teacher") or {}
  payload = {
    "split": "prove",
    "n_seqs": 10,
    "reference": "teacher_y_clean",
    "eval_mode": "merged_LunionR",
    "image_2d": {
      "box_source": "detector_preds",
      "not_hot3d_gt_boxes": True,
      **_round3(prove.get("image_2d") or {}, ("frame_prec", "frame_f1", "seg_prec", "seg_f1")),
      "source": "outputs/baseline/image_2d_det_prove_merged/prove_summary_vs_teacher.json",
    },
    "gru_phase_d_r7": {
      **_round3(prove.get("gru_phase_d_r7") or {}, ("frame_prec", "frame_f1", "seg_prec", "seg_f1")),
      "source": "outputs/student/eval_prove_phase_d_r7_merged/prove_summary.json",
    },
    "invalid_do_not_use": {
      "note": "Prior image_2d_prove used HOT3D GT box2d — oracle boxes, not field-fair",
      "path": "outputs/baseline/image_2d_prove/",
      "mean_frame_f1": 0.701,
      "mean_seg_f1": 0.509,
    },
  }
  path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Aggregate merged L∪R results tables")
  p.add_argument(
    "--output",
    type=Path,
    default=EGOUSE_ROOT / "outputs" / "plots" / "final" / "results_table.md",
  )
  p.add_argument(
    "--json-out",
    type=Path,
    default=EGOUSE_ROOT / "outputs" / "plots" / "final" / "results_p0003_c701bd11.json",
  )
  p.add_argument(
    "--prove-json-out",
    type=Path,
    default=EGOUSE_ROOT / "outputs" / "plots" / "final" / "results_prove_vs_teacher.json",
  )
  return p.parse_args()


def main() -> None:
  args = parse_args()
  data = collect_metrics()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(render_markdown(data) + "\n")
  args.json_out.write_text(json.dumps(data, indent=2, default=str) + "\n")
  write_prove_json(data, args.prove_json_out)
  print(f"Wrote {args.output}")
  print(f"Wrote {args.json_out}")
  print(f"Wrote {args.prove_json_out}")


if __name__ == "__main__":
  main()
