import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from lib.paths import setup_paths

setup_paths()

from lib.viz.timeline import (
  RULE_PLOT_SPECS,
  build_timeline_table,
  load_scoring_config,
  plot_gt_timeline_figure,
  plot_per_object_hoi_figure,
  plot_rule_timeline_figure,
)

frame_scores_path = _ROOT / "outputs" / "interactions" / "frame_scores.csv"
ground_truths_path = _ROOT / "src" / "data" / "ground_truth" / "P0003_c701bd11_segments.csv"
mapping_path = _ROOT / "hot3d_data" / "P0003_c701bd11" / "timecode_devicetime_mapping.csv"
scoring_config_path = _ROOT / "outputs" / "interactions" / "scoring_config.json"
output_dir = _ROOT / "outputs" / "plots"


def parse_arguments():
  parser = argparse.ArgumentParser(description="Plot timeline of interactions")
  parser.add_argument("--frame_scores_path", type=str, default=frame_scores_path)
  parser.add_argument(
    "--pair_scores_path",
    type=str,
    default=None,
    help="Optional pair_scores.csv for per-object HOI swimlanes",
  )
  parser.add_argument("--ground_truths_path", type=str, default=ground_truths_path)
  parser.add_argument("--mapping_path", type=str, default=mapping_path)
  parser.add_argument("--scoring_config_path", type=str, default=scoring_config_path)
  parser.add_argument("--output_dir", type=str, default=output_dir)
  parser.add_argument(
    "--rule",
    choices=["gt", *RULE_PLOT_SPECS.keys(), "all", "per_object"],
    default="all",
    help="Which signal to plot (per_object=HOI swimlanes from pair_scores)",
  )
  parser.add_argument(
    "--layout",
    choices=("stacked", "overlay"),
    default="stacked",
    help="stacked=GT row + rule row per hand; overlay=GT under curves on one row",
  )
  parser.add_argument("--smooth_window", type=int, default=5, help="Rolling mean window (frames)")
  parser.add_argument(
    "--gap_sec",
    type=float,
    default=0.5,
    help="Max OFF gap to forgive between ON runs",
  )
  parser.add_argument(
    "--dwell_sec",
    type=float,
    default=0.3,
    help="Min ON duration (s) for processed regions",
  )
  parser.add_argument(
    "--tick_sec",
    type=float,
    default=10.0,
    help="Major x-axis tick spacing in seconds (default 10; use 5 for finer)",
  )
  return parser.parse_args()


def main():
  args = parse_arguments()
  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  timeline_df = build_timeline_table(
    Path(args.frame_scores_path),
    Path(args.ground_truths_path),
    Path(args.mapping_path),
  )
  scoring_config = load_scoring_config(Path(args.scoring_config_path))
  gap_sec = args.gap_sec

  pair_path = args.pair_scores_path
  if pair_path is None:
    candidate = Path(args.frame_scores_path).with_name("pair_scores.csv")
    if candidate.is_file():
      pair_path = str(candidate)

  if args.rule in ("gt", "all"):
    plot_gt_timeline_figure(
      timeline_df,
      tick_sec=args.tick_sec,
      output_path=out_dir / "timeline_gt.png",
    )
    print(f"Wrote {out_dir / 'timeline_gt.png'}")

  if args.rule in ("per_object", "all") and pair_path is not None:
    pair_df = pd.read_csv(pair_path)
    path = out_dir / "timeline_per_object.png"
    plot_per_object_hoi_figure(
      pair_df,
      timeline_df,
      tick_sec=args.tick_sec,
      output_path=path,
    )
    print(f"Wrote {path}")
  elif args.rule == "per_object":
    raise SystemExit("--rule per_object requires pair_scores.csv (pass --pair_scores_path)")

  rules = list(RULE_PLOT_SPECS.keys()) if args.rule == "all" else [args.rule]
  if args.rule not in ("gt", "per_object"):
    for rule in rules:
      if rule not in RULE_PLOT_SPECS:
        continue
      spec = RULE_PLOT_SPECS[rule]
      if spec["score_col"] not in timeline_df.columns:
        print(f"Skip {rule}: missing column {spec['score_col']}")
        continue
      if spec["threshold_key"] not in scoring_config:
        print(f"Skip {rule}: missing config key {spec['threshold_key']}")
        continue
      path = out_dir / f"timeline_{rule}.png"
      plot_rule_timeline_figure(
        timeline_df,
        rule,
        scoring_config=scoring_config,
        smooth_window=args.smooth_window,
        gap_sec=gap_sec,
        dwell_sec=args.dwell_sec,
        tick_sec=args.tick_sec,
        layout=args.layout,
        output_path=path,
      )
      print(f"Wrote {path}")

  plt.close("all")


if __name__ == "__main__":
  main()
