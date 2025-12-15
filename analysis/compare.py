import numpy as np
from typing import Dict, Any, Tuple, Optional
from . import thd


def _to_mono(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr if arr.ndim == 1 else arr[:, 0]


def _smooth_abs(x: np.ndarray, win: int = 256) -> np.ndarray:
    mag = np.abs(x)
    if len(mag) < win:
        return mag
    kernel = np.ones(win) / float(win)
    return np.convolve(mag, kernel, mode='same')


def align_signals(
    ref: np.ndarray,
    target: np.ndarray,
    max_lag_samples: Optional[int] = None,
    prefer_onset: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Align signals using envelope cross-correlation with optional lag bounds."""

    ref_raw = _to_mono(ref)
    tgt_raw = _to_mono(target)
    ref_mono = _smooth_abs(ref_raw)
    tgt_mono = _smooth_abs(tgt_raw)

    corr = np.correlate(tgt_mono, ref_mono, mode='full')
    center = len(ref_mono) - 1
    lag_corr = int(np.argmax(corr) - center)
    if max_lag_samples is not None:
        window = slice(max(0, center - max_lag_samples), min(len(corr), center + max_lag_samples + 1))
        subcorr = corr[window]
        lag_corr = int(np.argmax(subcorr) + window.start - center)

    def onset_idx(raw: np.ndarray) -> int:
        abs_raw = np.abs(raw)
        thresh = 0.1 * float(np.max(abs_raw) + 1e-12)
        idxs = np.nonzero(abs_raw > thresh)[0]
        return int(idxs[0]) if idxs.size else 0

    lag_onset = onset_idx(tgt_raw) - onset_idx(ref_raw)
    lag = lag_onset if prefer_onset and lag_onset != 0 else lag_corr

    if lag >= 0:
        aligned_ref = ref[: len(ref) - lag]
        aligned_tgt = target[lag : lag + len(aligned_ref)]
    else:
        aligned_ref = ref[-lag : -lag + len(target)]
        aligned_tgt = target[: len(aligned_ref)]

    min_len = min(len(aligned_ref), len(aligned_tgt))
    return aligned_ref[:min_len], aligned_tgt[:min_len], lag


def gain_match(
    ref: np.ndarray,
    target: np.ndarray,
    stable_region: Tuple[float, float] = (0.05, 0.95),
) -> Tuple[np.ndarray, float]:
    ref_mono = _to_mono(ref)
    tgt_mono = _to_mono(target)
    n = len(ref_mono)
    s, e = stable_region
    s_idx = int(n * s)
    e_idx = int(n * e)
    if e_idx <= s_idx:
        s_idx, e_idx = 0, n
    ref_slice = ref_mono[s_idx:e_idx]
    tgt_slice = tgt_mono[s_idx:e_idx]
    rms_ref = np.sqrt(np.mean(ref_slice ** 2) + 1e-12)
    rms_tgt = np.sqrt(np.mean(tgt_slice ** 2) + 1e-12)
    gain = rms_ref / max(rms_tgt, 1e-12)
    gain_db = 20 * np.log10(1.0 / gain + 1e-12)
    return target * gain, gain_db


def _freq_response_delta(ref: np.ndarray, tgt: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = np.hanning(len(ref))
    spec_ref = np.fft.rfft(ref * window)
    spec_tgt = np.fft.rfft(tgt * window)
    mag_ref = 20 * np.log10(np.abs(spec_ref) + 1e-12)
    mag_tgt = 20 * np.log10(np.abs(spec_tgt) + 1e-12)
    freqs = np.fft.rfftfreq(len(ref), 1 / fs)
    return freqs, mag_ref, mag_tgt


def _detect_hum_peaks(freqs: np.ndarray, mag: np.ndarray) -> list:
    hum_bins = []
    for base in (50, 60):
        for mul in range(1, 6):
            f = base * mul
            idx = np.argmin(np.abs(freqs - f))
            hum_bins.append({"freq": float(f), "level_db": float(mag[idx])})
    return hum_bins


def residual_metrics(
    ref: np.ndarray,
    tgt: np.ndarray,
    fs: int,
    freq: float,
    hmax: int = 5,
    stable_region: Tuple[float, float] = (0.05, 0.95),
    include_residual: bool = False,
) -> Dict[str, Any]:
    n = min(len(ref), len(tgt))
    ref = ref[:n]
    tgt = tgt[:n]

    s, e = stable_region
    s_idx = int(n * s)
    e_idx = int(n * e) if int(n * e) > s_idx else n

    ref_core = ref[s_idx:e_idx]
    tgt_core = tgt[s_idx:e_idx]
    residual = tgt_core - ref_core

    res_rms = np.sqrt(np.mean(residual ** 2) + 1e-12)
    ref_rms = np.sqrt(np.mean(ref_core ** 2) + 1e-12)
    snr = 20 * np.log10(ref_rms / res_rms + 1e-12)
    noise_floor = 20 * np.log10(res_rms + 1e-12)

    thd_ref = thd.compute_thd(ref_core, fs, freq, hmax)
    thd_tgt = thd.compute_thd(tgt_core, fs, freq, hmax)
    thd_delta = thd_tgt["thd_db"] - thd_ref["thd_db"]

    freqs, mag_ref, mag_tgt = _freq_response_delta(ref_core, tgt_core, fs)
    fr_dev = mag_tgt - mag_ref
    band = (freqs >= 20) & (freqs <= 20000)
    fr_band = fr_dev[band]
    fr_dev_median = float(np.median(fr_band)) if fr_band.size else float(np.median(fr_dev))
    fr_dev_max = float(np.max(np.abs(fr_band))) if fr_band.size else float(np.max(np.abs(fr_dev)))

    hum_peaks = _detect_hum_peaks(freqs, mag_tgt)

    clipping = int(np.sum(np.abs(tgt_core) >= 0.999))

    metrics: Dict[str, Any] = {
        "residual_rms_dbfs": 20 * np.log10(res_rms + 1e-12),
        "snr_db": snr,
        "noise_floor_dbfs": noise_floor,
        "thd_ref_db": thd_ref["thd_db"],
        "thd_tgt_db": thd_tgt["thd_db"],
        "thd_delta_db": thd_delta,
        "fr_dev_median_db": fr_dev_median,
        "fr_dev_max_db": fr_dev_max,
        "hum_peaks": hum_peaks,
        "clipping_samples": clipping,
    }
    if include_residual:
        metrics["residual"] = residual
    return metrics


