from pathlib import Path
import pandas as pd
import numpy as np

def load_time_mapping(mapping_path: Path) -> pd.DataFrame:
    """Load the timecode_devicetime_mapping.csv file and return a dataframe."""
    return pd.read_csv(mapping_path)


def load_segments(segment_path: Path) -> pd.DataFrame:
    """Load the ground truth segments from the segments.csv file."""
    segments_df =pd.read_csv(segment_path)
    segments_df["start_devicetime_ns"] = pd.to_numeric(segments_df["start_devicetime_ns"]).astype('Int64')
    segments_df["end_devicetime_ns"] = pd.to_numeric(segments_df["end_devicetime_ns"]).astype('Int64')
    return segments_df


def validate_segments(segments_df: pd.DataFrame) -> None:
    """Validate the segments dataframe."""
    for hand_side, hand_df in segments_df.groupby("hand_side"):
        hand_df = hand_df.sort_values(by="start_devicetime_ns")
        hand_df = hand_df.reset_index(drop=True)
        for i in range(len(hand_df) - 1):
            prev = hand_df.iloc[i]
            next = hand_df.iloc[i + 1]
            if prev["end_devicetime_ns"] > next["start_devicetime_ns"]:
                raise ValueError(f"Segments for {hand_side} are not valid: {prev['end_devicetime_ns']} > {next['start_devicetime_ns']}")


def device_to_timecode(device_time_ns: int, mapping: pd.DataFrame) -> int:
    """Convert a device time to a timecode."""
    dev = mapping["devicetime_ns"].to_numpy()
    tc = mapping["timecode_ns"].to_numpy()
    idx = np.searchsorted(dev, device_time_ns)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(dev):
        candidates.append(idx)
    best_idx = min(candidates, key=lambda x: abs(dev[x] - device_time_ns))
    timecode = int(tc[best_idx])
    return timecode


def timecode_to_device(timecode_ns: int, mapping: pd.DataFrame) -> int:
    """Convert a timecode to the nearest device time."""
    tc = mapping["timecode_ns"].to_numpy()
    dev = mapping["devicetime_ns"].to_numpy()
    idx = np.searchsorted(tc, timecode_ns)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(tc):
        candidates.append(idx)
    best_idx = min(candidates, key=lambda x: abs(tc[x] - timecode_ns))
    return int(dev[best_idx])


def convert_segment_boundaries(segments_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Add start_timecode and end_timecode columns to the segments dataframe."""
    copy_segments_df = segments_df.copy()
    copy_segments_df["start_timecode_ns"] = copy_segments_df["start_devicetime_ns"].apply(lambda x: device_to_timecode(x, mapping_df))
    copy_segments_df["end_timecode_ns"] = copy_segments_df["end_devicetime_ns"].apply(lambda x: device_to_timecode(x, mapping_df))
    return copy_segments_df


def label_frames(segments_df: pd.DataFrame, frame_scores_df: pd.DataFrame) -> pd.DataFrame:
    """Label frames based on the ground truth segments."""
    rows = []
    for _, row in frame_scores_df.iterrows():
        frame_timecode_ns = row["timestamp_ns"]
        frame_hand_side = row["hand_side"]
        gt_object_name = None
        gt_interacting = True
        hand_segments = segments_df[segments_df["hand_side"] == frame_hand_side]
        for _, segment_row in hand_segments.iterrows():
            if frame_timecode_ns >= segment_row["start_timecode_ns"] and frame_timecode_ns <= segment_row["end_timecode_ns"]:
                gt_object_name = segment_row["object_name"]
                break
        if gt_object_name is None:
            gt_interacting = False
            gt_object_name = "none"
        rows.append({
            "timestamp_ns": frame_timecode_ns,
            "hand_side": frame_hand_side,
            "gt_object_name": gt_object_name,
            "gt_interacting": gt_interacting,
        })
    return pd.DataFrame(rows)


def confusion_binary(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    """Create a confusion matrix for binary classification."""
    return {
        "tp": ((y_true) & (y_pred)).sum(),
        "fp": ((~y_true) & (y_pred)).sum(),
        "fn": ((y_true) & (~y_pred)).sum(),
        "tn": ((~y_true) & (~y_pred)).sum(),
    }


def strict_confusion_binary(y_true: pd.Series, y_pred: pd.Series, gt_obj: pd.Series, pred_obj: pd.Series) -> dict[str, int]:
    """Create a confusion matrix for binary classification with strict object matching."""
    obj_matches = (gt_obj == pred_obj)
    tp = (obj_matches & y_true & y_pred).sum()
    fp = ((~y_true & y_pred) | (~obj_matches & y_pred & y_true)).sum()
    fn = ((y_true) & (~y_pred)).sum()
    tn = ((~y_true) & (~y_pred)).sum()
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def round_metric(x, ndigits=3) -> float:
    return None if x is None else round(x, ndigits)


def calculate_metrics(confusion: dict[str, int]) -> dict[str, float]:
    """Calculate the metrics for a confusion matrix."""
    tp, fp, fn, tn = confusion["tp"], confusion["fp"], confusion["fn"], confusion["tn"]
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None else None
    f0_5 = 1.25 * precision * recall / (0.25 * precision + recall) if precision is not None and recall is not None else None
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn > 0 else None
    return {
        "prec": round_metric(precision),
        "rec": round_metric(recall),
        "f1": round_metric(f1),
        "f0_5": round_metric(f0_5),
        "acc": round_metric(accuracy),
    }


def evaluate_rules(frame_score_df: pd.DataFrame, labeled_df: pd.DataFrame, rule_columns: list[dict[str, str]]) -> pd.DataFrame:
    """Evaluate the rules from the frame_scores based on the labeled dataframe."""
    rows = []
    copy_frame_score_df = frame_score_df.copy()
    copy_labeled_df = labeled_df.copy()
    for df in (copy_frame_score_df, copy_labeled_df):
        df["timestamp_ns"] = df["timestamp_ns"].astype("int64")
        df["hand_side"] = df["hand_side"].astype(str)
    merged = copy_frame_score_df.merge(copy_labeled_df, on=["timestamp_ns", "hand_side"], how="inner")
    if len(merged) != len(copy_frame_score_df):
        raise ValueError(f"Merged dataframe has {len(merged)} rows, but {len(copy_frame_score_df)} rows in the frame score dataframe")
    y_true = merged["gt_interacting"].astype(bool)
    gt_obj = merged["gt_object_name"].astype(str)
    for rule_column in rule_columns:
        y_pred = merged[rule_column["interacting_col"]].astype(bool)
        pred_obj = merged[rule_column["object_col"]].astype(str)
        confusion = confusion_binary(y_true, y_pred)
        metrics = calculate_metrics(confusion)
        strict_confusion = strict_confusion_binary(y_true, y_pred, gt_obj, pred_obj)
        strict_metrics = calculate_metrics(strict_confusion)
        rows.append({
            "rule": rule_column["name"],
            **confusion,
            **metrics,
            "s_tp": strict_confusion["tp"],
            "s_fp": strict_confusion["fp"],
            "s_fn": strict_confusion["fn"],
            "s_tn": strict_confusion["tn"],
            "s_prec": strict_metrics["prec"],
            "s_rec": strict_metrics["rec"],
            "s_f1": strict_metrics["f1"],
            "s_f0_5": strict_metrics["f0_5"],
            "s_acc": strict_metrics["acc"],
        })
    return pd.DataFrame(rows)