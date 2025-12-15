import numpy as np
from typing import Dict, Any, Tuple
from . import thd


def _smooth_abs(x: np.ndarray, win: int = 256) -> np.ndarray:
    mag = np.abs(x)
    if len(mag) < win:
        return mag
    kernel = np.ones(win) / float(win)
    return np.convolve(mag, kernel, mode='same')


def align_signals(ref: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    ref_mono = ref if ref.ndim == 1 else ref[:, 0]
    tgt_mono = target if target.ndim == 1 else target[:, 0]

    ref_env = _smooth_abs(ref_mono)
    tgt_env = _smooth_abs(tgt_mono)

    pad = len(ref_env)
    ref_pad = np.concatenate([ref_env, np.zeros(pad)])
    tgt_pad = np.concatenate([tgt_env, np.zeros(pad)])

    corr = np.correlate(tgt_pad, ref_pad, mode='full')
    lag_corr = int(np.argmax(corr) - (len(ref_pad) - 1))

    # Onset-based estimate to stabilize noisy cross-correlations using raw magnitude
    def onset_idx(raw: np.ndarray) -> int:
        abs_raw = np.abs(raw)
        thresh = 0.1 * float(np.max(abs_raw) + 1e-12)
        idxs = np.nonzero(abs_raw > thresh)[0]
        return int(idxs[0]) if idxs.size else 0

    lag_onset = onset_idx(tgt_mono) - onset_idx(ref_mono)
    lag = lag_onset if lag_onset != 0 else lag_corr

    if lag >= 0:
        aligned_ref = ref_pad[: len(ref_pad) - lag]
        aligned_tgt = tgt_pad[lag : lag + len(aligned_ref)]
    else:
        aligned_tgt = tgt_pad[: len(tgt_pad) + lag]
        aligned_ref = ref_pad[-lag : -lag + len(aligned_tgt)]

    min_len = min(len(aligned_ref), len(aligned_tgt))
    return aligned_ref[:min_len], aligned_tgt[:min_len], lag


def gain_match(ref: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    if ref.ndim > 1:
        ref_mono = ref[:, 0]
    else:
        ref_mono = ref
    if target.ndim > 1:
        tgt_mono = target[:, 0]
    else:
        tgt_mono = target
    rms_ref = np.sqrt(np.mean(ref_mono ** 2) + 1e-12)
    rms_tgt = np.sqrt(np.mean(tgt_mono ** 2) + 1e-12)
    gain = rms_ref / max(rms_tgt, 1e-12)
    return target * gain, 20 * np.log10(1.0 / gain + 1e-12)


def residual_metrics(ref: np.ndarray, tgt: np.ndarray, fs: int, freq: float, hmax: int = 5) -> Dict[str, Any]:
    residual = tgt - ref
    res_rms = np.sqrt(np.mean(residual ** 2) + 1e-12)
    ref_rms = np.sqrt(np.mean(ref ** 2) + 1e-12)
    snr = 20 * np.log10(ref_rms / res_rms + 1e-12)
    noise_floor = 20 * np.log10(res_rms + 1e-12)
    thd_ref = thd.compute_thd(ref, fs, freq, hmax)
    thd_tgt = thd.compute_thd(tgt, fs, freq, hmax)
    thd_delta = thd_tgt['thd_db'] - thd_ref['thd_db']
    window = np.hanning(len(ref))
    spec_ref = np.fft.rfft(ref * window)
    spec_tgt = np.fft.rfft(tgt * window)
    mag_ref = 20 * np.log10(np.abs(spec_ref) + 1e-12)
    mag_tgt = 20 * np.log10(np.abs(spec_tgt) + 1e-12)
    fr_dev = mag_tgt - mag_ref
    fr_mean_dev = float(np.median(fr_dev))
    hum_bins = []
    freqs = np.fft.rfftfreq(len(ref), 1 / fs)
    for base in (50, 60):
        for mul in range(1, 5):
            f = base * mul
            idx = np.argmin(np.abs(freqs - f))
            hum_bins.append({'freq': f, 'level_db': float(mag_tgt[idx])})
    return {
        'residual_rms_dbfs': 20 * np.log10(res_rms + 1e-12),
        'snr_db': snr,
        'noise_floor_dbfs': noise_floor,
        'thd_ref_db': thd_ref['thd_db'],
        'thd_tgt_db': thd_tgt['thd_db'],
        'thd_delta_db': thd_delta,
        'fr_dev_median_db': fr_mean_dev,
        'hum_peaks': hum_bins,
        'residual': residual,
    }


def compare_signals(ref: np.ndarray, target: np.ndarray, fs: int, freq: float, hmax: int = 5) -> Dict[str, Any]:
    """Align, gain-match, and measure residual metrics between two signals."""

    aligned_ref, aligned_tgt, lag = align_signals(ref, target)
    gain_matched, gain_error_db = gain_match(aligned_ref, aligned_tgt)
    metrics = residual_metrics(aligned_ref, gain_matched, fs, freq, hmax)

    latency_ms = lag / fs * 1000.0
    metrics.update(
        {
            'latency_samples': lag,
            'latency_ms': latency_ms,
            'gain_error_db': gain_error_db,
        }
    )
    return metrics
