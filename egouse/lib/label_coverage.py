"""Shared, strict reference knownness; missing coverage never means annotated."""
import numbers
import numpy as np
import pandas as pd


def known_mask(frame, label_column='v2_interacting'):
    labels = frame[label_column]
    if not labels.dropna().isin([0, 1]).all():
        raise ValueError('Reference labels must be binary or missing')
    if 'label_known' not in frame:
        return labels.notna()
    def parse(value):
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, numbers.Real) and value in (0, 1):
            return bool(value)
        raise ValueError(f'Invalid label_known value: {value!r}; expected boolean, 0/1, or missing')
    return frame.label_known.map(parse).astype(bool) & labels.notna()


def known_negative_false_positives(prediction, reference):
    from lib.evaluation_coverage import align_pairs
    aligned = align_pairs(prediction, reference)
    return aligned.loc[aligned.label_known & aligned.truth.eq(0) & aligned.prediction.eq(1)].copy()
