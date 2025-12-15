import numpy as np
from typing import Dict, Any


def _envelope_follow(x: np.ndarray, fs: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Simple envelope follower with attack/release time constants."""
    attack_coeff = float(np.exp(-1.0 / (max(attack_ms, 1e-6) / 1000.0 * fs)))
    release_coeff = float(np.exp(-1.0 / (max(release_ms, 1e-6) / 1000.0 * fs)))
    env = np.zeros_like(x, dtype=np.float32)
    last = 0.0
    for i, sample in enumerate(np.abs(x)):
        if sample > last:
            coeff = attack_coeff
        else:
            coeff = release_coeff
        last = coeff * last + (1.0 - coeff) * sample
        env[i] = last
    return env


def _soft_knee_gain(level_db: float, threshold_db: float, ratio: float, knee_db: float) -> float:
    """Gain computer implementing a soft knee if knee_db > 0."""
    if knee_db <= 0:
        if level_db <= threshold_db:
            return 0.0
        compressed = threshold_db + (level_db - threshold_db) / max(ratio, 1e-12)
        return compressed - level_db

    lower = threshold_db - knee_db / 2.0
    upper = threshold_db + knee_db / 2.0
    if level_db < lower:
        return 0.0
    if level_db > upper:
        compressed = threshold_db + (level_db - threshold_db) / max(ratio, 1e-12)
        return compressed - level_db
    # Within knee region: quadratic interpolation for smooth transition
    delta = level_db - lower
    compressed = level_db + (1.0 / max(ratio, 1e-12) - 1.0) * (delta ** 2) / (2.0 * knee_db)
    return compressed - level_db


def apply_compressor(
    x: np.ndarray,
    threshold_db: float,
    ratio: float,
    makeup_db: float,
    knee_db: float = 0.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    fs: int = 48000,
) -> np.ndarray:
    """Apply a simple feed-forward compressor to ``x``.

    The implementation uses an RMS-like detector with separate attack/release
    coefficients, an optional soft knee, and linear makeup gain. This keeps the
    public API stable while allowing the estimator tests to exercise the gain
    computer reliably.
    """

    if x.ndim > 1:
        x_mono = x[:, 0]
    else:
        x_mono = x

    env = _envelope_follow(x_mono, fs, attack_ms, release_ms)
    out = np.zeros_like(x_mono, dtype=np.float32)

    for i, sample in enumerate(x_mono):
        level = max(env[i], 1e-12)
        level_db = 20.0 * np.log10(level)
        gain_db = _soft_knee_gain(level_db, threshold_db, ratio, knee_db)
        total_gain_db = gain_db + makeup_db
        lin_gain = 10.0 ** (total_gain_db / 20.0)
        out[i] = float(sample * lin_gain)

    return out if x.ndim == 1 else out[:, None]


def build_stepped_tone(freq: float, fs: int, amp_max: float = 1.36) -> Dict[str, Any]:
    seg_dur, gap_dur = 0.25, 0.05
    amps = np.linspace(0.05, amp_max, 36)
    protect = amp_max
    t_seg = np.linspace(0, seg_dur, int(fs * seg_dur), endpoint=False)
    gap = np.zeros(int(fs * gap_dur))
    tx = np.concatenate([
        np.concatenate((min(a, protect) * np.sin(2 * np.pi * freq * t_seg), gap)) for a in amps
    ])
    meta = {
        'seg_samples': int(seg_dur * fs),
        'gap_samples': int(gap_dur * fs),
        'amps': amps,
        'trim_lead': int(0.03 * fs),
        'trim_tail': int(0.01 * fs),
    }
    return {'signal': tx.astype(np.float32), 'meta': meta}


def compression_curve(sig: np.ndarray, meta: Dict[str, Any], fs: int, freq: float) -> Dict[str, Any]:
    segN = meta['seg_samples']
    gapN = meta['gap_samples']
    amps = meta['amps']
    trim_lead = meta.get('trim_lead', int(0.03 * fs))
    trim_tail = meta.get('trim_tail', int(0.01 * fs))

    rms_in_db, rms_out_db = [], []
    for A, i in zip(amps, range(len(amps))):
        s0 = i * (segN + gapN)
        s1 = s0 + segN
        seg = sig[s0:s1]
        seg = seg[trim_lead:max(trim_lead, len(seg) - trim_tail)]
        rin = max(A / np.sqrt(2), 1e-12)
        rout = max(np.sqrt(np.mean(np.square(seg))), 1e-12)
        rms_in_db.append(20 * np.log10(rin))
        rms_out_db.append(20 * np.log10(rout))
    rms_in_db = np.array(rms_in_db)
    rms_out_db = np.array(rms_out_db)
    diff = rms_out_db - rms_in_db

    a_all, b_all = np.polyfit(rms_in_db, rms_out_db, 1)
    gain_offset_db = float(np.mean(diff))
    slope_tol, spread_tol = 0.05, 1.0
    no_compression = (abs(a_all - 1.0) < slope_tol) and ((diff.max() - diff.min()) < spread_tol)

    if no_compression:
        thr, ratio = np.nan, 1.0
    else:
        mask = diff < -0.5
        if np.count_nonzero(mask) < 2:
            thr, ratio = np.nan, 1.0
            no_compression = True
        else:
            x, y = rms_in_db[mask], rms_out_db[mask]
            a, b = np.polyfit(x, y, 1)
            ratio = 1.0 / max(a, 1e-12)
            thr = b / (1 - a) if abs(1 - a) > 1e-6 else np.nan
    return {
        'in_db': rms_in_db,
        'out_db': rms_out_db,
        'gain_offset_db': gain_offset_db,
        'no_compression': no_compression,
        'thr_db': thr,
        'ratio': ratio,
    }
