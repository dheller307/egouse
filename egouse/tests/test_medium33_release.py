"""Tests for the maintained release decoder, export boundary and asset paths."""

import json

import numpy as np

from release.cli import F, VOCAB, decode, intervals
from release.config import ASSET_KEYS, user_paths


def test_release_decoder_carries_pairs_and_resets_only_at_declared_gaps():
    clock = np.array([0, 30_000_000, 60_000_000, 180_000_000, 210_000_000])
    prob = np.zeros((2, len(clock), len(VOCAB)), dtype=np.float32)
    aria = VOCAB.index('aria_small')
    assert aria == 32
    prob[0, :, aria] = [.50, .47, .47, .47, .50]
    expected = decode(prob, clock)
    assert expected[0, :, aria].tolist() == [True, True, True, False, True]
    assert not expected[1].any()
    np.testing.assert_array_equal(decode(prob, clock, chunk=2), expected)
    assert not decode(prob[:, 1:2], clock[1:2]).any()


def test_half_open_export_splits_at_clock_gap_and_keeps_canonical_identity():
    clock = np.array([100, 30_000_100, 60_000_100, 180_000_100, 210_000_100])
    on = np.zeros((len(clock), len(VOCAB)), dtype=bool)
    on[:, 32] = True
    spans = intervals(on, clock, 'merged')
    assert [(s['start_frame'], s['end_frame_exclusive']) for s in spans] == [(0, 3), (3, 5)]
    assert [(s['start_ns'], s['end_ns']) for s in spans] == [(100, 90_000_100), (180_000_100, 240_000_100)]
    assert all(s['hand'] == 'merged' and s['class_id'] == 32 and s['object'] == 'aria_small' for s in spans)
    assert F['decoder']['clock_gap_ns'] == 100_000_000


def test_local_asset_config_resolves_relative_to_its_own_directory(tmp_path):
    location = tmp_path / 'assets.local.json'
    location.write_text(json.dumps({k: f'bin/{k}' for k in ASSET_KEYS}))
    assert user_paths(location) == {k: tmp_path / 'bin' / k for k in ASSET_KEYS}
