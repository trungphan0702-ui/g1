import numpy as np

from analysis import compare


def _make_step(delay_samples: int, length: int = 200):
    sig = np.zeros(length)
    sig[20:] = 1.0
    if delay_samples >= 0:
        padded = np.concatenate([np.zeros(delay_samples), sig])
        return sig, padded[: length + delay_samples + 50]
    # negative delay: target starts earlier
    padded = np.concatenate([sig[-delay_samples:], np.zeros(-delay_samples)])
    return sig, padded


def test_align_signals_zero_lag():
    ref, tgt = _make_step(0)
    a_ref, a_tgt, lag = compare.align_signals(ref, tgt, max_lag_samples=100)
    assert lag == 0
    assert np.allclose(a_ref, a_tgt)


def test_align_signals_positive_lag():
    ref, tgt = _make_step(12)
    a_ref, a_tgt, lag = compare.align_signals(ref, tgt, max_lag_samples=50)
    assert lag == 12
    assert len(a_ref) == len(a_tgt)
    assert np.allclose(a_ref, a_tgt)


def test_align_signals_negative_lag():
    ref, tgt = _make_step(-8)
    a_ref, a_tgt, lag = compare.align_signals(ref, tgt, max_lag_samples=50)
    assert lag == -8
    assert len(a_ref) == len(a_tgt)
    assert np.allclose(a_ref, a_tgt)


def test_align_handles_length_mismatch():
    ref = np.ones(1000)
    tgt = np.concatenate([np.zeros(20), np.ones(800)])
    a_ref, a_tgt, lag = compare.align_signals(ref, tgt, max_lag_samples=200)
    assert lag == 20
    assert len(a_ref) == len(a_tgt)
    # gain match expectation: overlap region equals ones
    assert np.allclose(a_ref, 1)
    assert np.allclose(a_tgt, 1)


def test_align_prefers_onset_detection():
    ref = np.concatenate([np.zeros(10), np.ones(30)])
    tgt = np.concatenate([np.zeros(22), np.ones(30)])
    a_ref, a_tgt, lag = compare.align_signals(ref, tgt, prefer_onset=True, max_lag_samples=50)
    assert lag == 12
    assert np.allclose(a_ref, a_tgt)
