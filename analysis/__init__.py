"""Single-file analysis package bundling previous submodules.

This module dynamically creates lightweight submodules so existing import
patterns (``from analysis import thd`` or ``import analysis.live_measurements``)
continue to work while keeping all code in one file.
"""
import os
import sys
import types
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Utility helpers -------------------------------------------------------------

def _create_submodule(name: str) -> types.ModuleType:
    mod = types.ModuleType(__name__ + "." + name)
    sys.modules[mod.__name__] = mod
    setattr(sys.modules[__name__], name, mod)
    return mod


attack_release = _create_submodule("attack_release")
compare = _create_submodule("compare")
compressor = _create_submodule("compressor")
live_measurements = _create_submodule("live_measurements")
thd = _create_submodule("thd")

__all__ = ["attack_release", "compare", "compressor", "live_measurements", "thd"]

# --------------------------- attack_release ---------------------------------

def _generate_step_tone(freq: float, fs: int, amp: float = 0.7, duration: float = 2.0) -> np.ndarray:
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = amp * np.sin(2 * np.pi * freq * t)
    env = np.ones_like(tone) * 0.3
    attack_idx = len(env) // 4
    release_idx = 3 * len(env) // 4
    env[attack_idx:release_idx] = 1.0
    env[release_idx:] = 0.3
    return (tone * env).astype(np.float32)


def _envelope_rms(sig: np.ndarray, fs: int, win_ms: float) -> np.ndarray:
    win = int(max(1, fs * win_ms / 1000))
    if sig.ndim > 1:
        sig = sig[:, 0]
    padded = np.pad(sig ** 2, (win, win))
    cumsum = np.cumsum(padded)
    rms = np.sqrt((cumsum[2 * win:] - cumsum[:-2 * win]) / max(2 * win, 1))
    return rms


def _attack_release_times(sig: np.ndarray, fs: int, win_ms: float) -> Dict[str, float]:
    if sig.ndim > 1:
        sig = sig[:, 0]

    env = _envelope_rms(sig, fs, win_ms)
    if len(env) < 10 or not np.isfinite(np.max(env)):
        return {"attack_ms": float("nan"), "release_ms": float("nan")}

    n = len(env)
    q1, q3 = n // 4, 3 * n // 4
    low_level = float(np.median(env[:q1]))
    high_level = float(np.median(env[q1:q3]))
    tail_level = float(np.median(env[q3:]))

    peak = float(np.max(env))
    if peak < 1e-12:
        return {"attack_ms": float("nan"), "release_ms": float("nan")}

    atk_start_lvl = low_level + 0.1 * (high_level - low_level)
    atk_end_lvl = low_level + 0.9 * (high_level - low_level)
    rel_start_lvl = high_level - 0.1 * (high_level - tail_level)
    rel_end_lvl = high_level - 0.9 * (high_level - tail_level)

    search_rise = env[q1 - n // 10 : q3]
    search_fall = env[q3 - n // 10 :]

    def _crossing(x: np.ndarray, level: float, direction: str = "up") -> Optional[int]:
        idxs = np.nonzero(x >= level)[0] if direction == "up" else np.nonzero(x <= level)[0]
        return int(idxs[0]) if idxs.size else None

    atk_start_rel = _crossing(search_rise, atk_start_lvl, "up")
    atk_end_rel = _crossing(search_rise, atk_end_lvl, "up")
    rel_start_rel = _crossing(search_fall, rel_start_lvl, "down")
    rel_end_rel = _crossing(search_fall, rel_end_lvl, "down")

    attack_idx = atk_end_idx = release_idx = None
    if atk_start_rel is not None and atk_end_rel is not None:
        attack_idx = (q1 - n // 10) + atk_start_rel
        atk_end_idx = (q1 - n // 10) + atk_end_rel
    if rel_start_rel is not None and rel_end_rel is not None:
        release_idx = (q3 - n // 10) + rel_end_rel
        rel_start_idx = (q3 - n // 10) + rel_start_rel
    else:
        rel_start_idx = None

    attack_ms = (atk_end_idx - attack_idx) / fs * 1000.0 if atk_end_idx is not None and attack_idx is not None else float("nan")
    release_ms = (release_idx - rel_start_idx) / fs * 1000.0 if release_idx is not None and rel_start_idx is not None else float("nan")

    if os.getenv("DSP_DEBUG"):
        print(
            f"[DSP_DEBUG][AR] low={low_level:.3e}, high={high_level:.3e}, tail={tail_level:.3e}, "
            f"atk_idx={attack_idx}, atk_end={atk_end_idx}, rel_start={rel_start_idx}, rel_end={release_idx}"
        )

    return {"attack_ms": attack_ms, "release_ms": release_ms}


def _compare_attack_release(input_sig: np.ndarray, output_sig: np.ndarray, fs: int, win_ms: float) -> Dict[str, Any]:
    in_times = _attack_release_times(input_sig, fs, win_ms)
    out_times = _attack_release_times(output_sig, fs, win_ms)
    return {
        "input": in_times,
        "output": out_times,
        "delta_attack": out_times["attack_ms"] - in_times["attack_ms"],
        "delta_release": out_times["release_ms"] - in_times["release_ms"],
    }

attack_release.generate_step_tone = _generate_step_tone
attack_release.envelope_rms = _envelope_rms
attack_release.attack_release_times = _attack_release_times
attack_release.compare_attack_release = _compare_attack_release

# ------------------------------- thd ----------------------------------------

def _normalize_thd_result(data: Dict[str, Any], fallback_db: float = 0.0) -> Dict[str, Any]:
    aliases = {
        "thdn": "thdn_db",
        "thdn_dB": "thdn_db",
        "thdn_db": "thdn_db",
        "thd+n_db": "thdn_db",
        "thd+n": "thdn_db",
        "thd_db": "thd_db",
        "thd": "thd_db",
    }
    normalized: Dict[str, Any] = dict(data)
    for key, target in aliases.items():
        if key in normalized and target not in normalized:
            normalized[target] = normalized[key]

    for key in ("thd_db", "thdn_db"):
        val = normalized.get(key, fallback_db)
        try:
            normalized[key] = float(val)
        except (TypeError, ValueError):
            normalized[key] = float(fallback_db)
    return normalized


def _compute_thd(
    signal: np.ndarray,
    fs: int,
    freq: float,
    max_h: int = 5,
    window: str = "hann",
    fundamental_band_bins: int = 2,
    nfft: Optional[int] = None,
) -> Dict[str, Any]:
    sig = np.asarray(signal, dtype=np.float32)
    sig = sig - np.mean(sig)
    if sig.ndim > 1:
        sig = sig[:, 0]

    nfft_use = int(nfft) if nfft else len(sig)
    if window == "hann" or window == "hanning":
        win = np.hanning(len(sig))
    elif window is None or window == "none":
        win = np.ones(len(sig))
    else:
        raise ValueError(f"Unsupported window '{window}'")

    windowed = sig * win
    spec = np.fft.rfft(windowed, n=nfft_use)
    freqs = np.fft.rfftfreq(nfft_use, 1 / fs)
    mag = np.abs(spec)
    power = mag ** 2
    if power.size:
        power[0] = 0.0

    fund_idx = int(np.argmin(np.abs(freqs - freq)))
    band_bins = int(max(1, fundamental_band_bins))
    band_start = max(fund_idx - band_bins, 1)
    band_stop = min(fund_idx + band_bins + 1, len(power))
    fund_band = slice(band_start, band_stop)
    fund_power = float(np.sum(power[fund_band]) + 1e-24)
    fund_mag = np.sqrt(fund_power)

    harmonics: Dict[int, float] = {}
    power_sum = 0.0
    for h in range(2, max_h + 1):
        idx = np.argmin(np.abs(freqs - h * freq))
        h_start = max(idx - band_bins, 1)
        h_stop = min(idx + band_bins + 1, len(power))
        h_power = float(np.sum(power[h_start:h_stop]))
        power_sum += h_power
        h_ratio = np.sqrt(h_power / fund_power) if fund_power > 0 else 0.0
        harmonics[h] = 20 * np.log10(h_ratio + 1e-12)

    thd_ratio = np.sqrt(power_sum / fund_power) if fund_power > 0 else 0.0
    thd_percent = thd_ratio * 100
    thd_db = 20 * np.log10(thd_ratio + 1e-12)

    power_total = float(np.sum(power))
    noise_power = max(power_total - fund_power, 0.0)
    thdn_ratio = np.sqrt(noise_power / fund_power) if fund_power > 0 else 0.0
    thdn_db = 20 * np.log10(thdn_ratio + 1e-12)

    result = {
        "fundamental_mag": fund_mag,
        "harmonics_dbc": harmonics,
        "thd_percent": thd_percent,
        "thd_ratio": thd_ratio,
        "thd_db": thd_db,
        "thdn_ratio": thdn_ratio,
        "thdn_db": thdn_db,
        "freqs": freqs,
        "spectrum": 20 * np.log10(mag + 1e-12),
        "fs": fs,
        "fund_freq": freq,
        "nfft": nfft_use,
        "window": "hann" if window == "hanning" else window,
        "fund_band_bins": band_bins,
    }

    normalized = _normalize_thd_result(result)

    if os.getenv("DSP_DEBUG"):
        print(
            f"[DSP_DEBUG] compute_thd: fund_idx={fund_idx}, fund_mag={fund_mag:.3e}, "
            f"thd_db={normalized['thd_db']:.2f}, thdn_db={normalized['thdn_db']:.2f}"
        )

    return normalized

thd.normalize_thd_result = _normalize_thd_result
thd.compute_thd = _compute_thd

# ------------------------------- compare ------------------------------------

def _to_mono(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr if arr.ndim == 1 else arr[:, 0]


def _smooth_abs(x: np.ndarray, win: int = 256) -> np.ndarray:
    mag = np.abs(x)
    if len(mag) < win:
        return mag
    kernel = np.ones(win) / float(win)
    return np.convolve(mag, kernel, mode="same")


def _align_signals(
    ref: np.ndarray,
    target: np.ndarray,
    max_lag_samples: Optional[int] = None,
    prefer_onset: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    ref_raw = _to_mono(ref)
    tgt_raw = _to_mono(target)
    ref_mono = _smooth_abs(ref_raw)
    tgt_mono = _smooth_abs(tgt_raw)

    corr = np.correlate(tgt_mono, ref_mono, mode="full")
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


def _gain_match(
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


def _residual_metrics(
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

    thd_ref = _compute_thd(ref_core, fs, freq, hmax)
    thd_tgt = _compute_thd(tgt_core, fs, freq, hmax)
    thd_delta = thd_tgt["thd_db"] - thd_ref["thd_db"]

    freqs = np.fft.rfftfreq(len(ref_core), 1 / fs)
    window = np.hanning(len(ref_core))
    spec_ref = np.fft.rfft(ref_core * window)
    spec_tgt = np.fft.rfft(tgt_core * window)
    mag_ref = 20 * np.log10(np.abs(spec_ref) + 1e-12)
    mag_tgt = 20 * np.log10(np.abs(spec_tgt) + 1e-12)
    fr_dev = mag_tgt - mag_ref
    band = (freqs >= 20) & (freqs <= 20000)
    fr_band = fr_dev[band]
    fr_dev_median = float(np.median(fr_band)) if fr_band.size else float(np.median(fr_dev))
    fr_dev_max = float(np.max(np.abs(fr_band))) if fr_band.size else float(np.max(np.abs(fr_dev)))

    hum_peaks = []
    for base in (50, 60):
        for mul in range(1, 6):
            f = base * mul
            idx = np.argmin(np.abs(freqs - f))
            hum_peaks.append({"freq": float(f), "level_db": float(mag_tgt[idx])})

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

compare._to_mono = _to_mono
compare._smooth_abs = _smooth_abs
compare.align_signals = _align_signals
compare.gain_match = _gain_match
compare.residual_metrics = _residual_metrics

# ------------------------------ compressor ----------------------------------

def _envelope_follow(x: np.ndarray, fs: int, attack_ms: float, release_ms: float) -> np.ndarray:
    attack_coeff = float(np.exp(-1.0 / (max(attack_ms, 1e-6) / 1000.0 * fs)))
    release_coeff = float(np.exp(-1.0 / (max(release_ms, 1e-6) / 1000.0 * fs)))
    env = np.zeros_like(x, dtype=np.float32)
    last = 0.0
    for i, sample in enumerate(np.abs(x)):
        coeff = attack_coeff if sample > last else release_coeff
        last = coeff * last + (1.0 - coeff) * sample
        env[i] = last
    return env


def _soft_knee_gain(level_db: float, threshold_db: float, ratio: float, knee_db: float) -> float:
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
    delta = level_db - lower
    compressed = level_db + (1.0 / max(ratio, 1e-12) - 1.0) * (delta ** 2) / (2.0 * knee_db)
    return compressed - level_db


def _apply_compressor(
    x: np.ndarray,
    threshold_db: float,
    ratio: float,
    makeup_db: float,
    knee_db: float = 0.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    fs: int = 48000,
) -> np.ndarray:
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


def _build_stepped_tone(freq: float, fs: int, amp_max: float = 1.36) -> Dict[str, Any]:
    seg_dur, gap_dur = 0.25, 0.05
    amps = np.linspace(0.05, amp_max, 36)
    protect = amp_max
    t_seg = np.linspace(0, seg_dur, int(fs * seg_dur), endpoint=False)
    gap = np.zeros(int(fs * gap_dur))
    tx = np.concatenate([
        np.concatenate((min(a, protect) * np.sin(2 * np.pi * freq * t_seg), gap)) for a in amps
    ])
    meta = {
        "seg_samples": int(seg_dur * fs),
        "gap_samples": int(gap_dur * fs),
        "amps": amps,
        "trim_lead": int(0.03 * fs),
        "trim_tail": int(0.01 * fs),
    }
    return {"signal": tx.astype(np.float32), "meta": meta}


def _compression_curve(sig: np.ndarray, meta: Dict[str, Any], fs: int, freq: float) -> Dict[str, Any]:
    segN = meta["seg_samples"]
    gapN = meta["gap_samples"]
    amps = meta["amps"]
    trim_lead = meta.get("trim_lead", int(0.03 * fs))
    trim_tail = meta.get("trim_tail", int(0.01 * fs))

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

    a_all, _ = np.polyfit(rms_in_db, rms_out_db, 1)
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
        "in_db": rms_in_db,
        "out_db": rms_out_db,
        "gain_offset_db": gain_offset_db,
        "no_compression": no_compression,
        "thr_db": thr,
        "ratio": ratio,
    }

compressor._envelope_follow = _envelope_follow
compressor._soft_knee_gain = _soft_knee_gain
compressor.apply_compressor = _apply_compressor
compressor.build_stepped_tone = _build_stepped_tone
compressor.compression_curve = _compression_curve

# --------------------------- live_measurements ------------------------------

def _lm_generate_thd_tone(freq: float, amp: float, fs: int, duration: float = 2.0) -> np.ndarray:
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sine = amp * np.sin(2 * np.pi * freq * t)
    return np.column_stack((sine, np.zeros_like(sine))).astype(np.float32)


def _lm_harmonic_metrics(signal: np.ndarray, fs: int, freq: float, max_h: int) -> Tuple[Dict[int, float], float, float]:
    n = len(signal)
    windowed = signal * np.hanning(n)
    fft = np.fft.rfft(windowed)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    fund_idx = int(np.argmin(np.abs(freqs - freq)))
    fund_mag = float(mag[fund_idx] + 1e-12)
    harmonics: Dict[int, float] = {}
    for h in range(2, max_h + 1):
        idx = int(np.argmin(np.abs(freqs - h * freq)))
        harmonics[h] = 20 * np.log10(float(mag[idx] + 1e-12) / fund_mag)
    thd_ratio = np.sqrt(np.sum([10 ** (v / 10.0) for v in harmonics.values()]))
    thd_percent = thd_ratio * 100.0
    thd_db = 20 * np.log10(thd_ratio + 1e-12)
    return harmonics, thd_percent, thd_db


def _lm_analyze_thd_capture(recorded: np.ndarray, fs: int, freq: float, hmax: int) -> Dict[str, Any]:
    trimmed = np.asarray(recorded, dtype=np.float32).flatten()
    trimmed = trimmed[int(0.05 * fs) :]
    peak = float(np.max(np.abs(trimmed)) + 1e-12)
    normalized = trimmed / peak
    harmonics, thd_percent, thd_db = _lm_harmonic_metrics(normalized, fs, freq, hmax)
    thd_metrics = _compute_thd(normalized, fs, freq, hmax)
    thd_metrics.update(
        {
            "harmonics_manual": harmonics,
            "thd_percent_manual": thd_percent,
            "thd_db_manual": thd_db,
            "normalized_signal": normalized,
        }
    )
    return thd_metrics


def _lm_generate_compressor_tone(freq: float, fs: int, amp_max: float = 1.36) -> Dict[str, Any]:
    seg_dur, gap_dur = 0.25, 0.05
    amps = np.linspace(0.05, amp_max, 36)
    protect = amp_max
    t_seg = np.linspace(0, seg_dur, int(fs * seg_dur), endpoint=False)
    gap = np.zeros(int(fs * gap_dur))
    tx = np.concatenate([np.concatenate((min(a, protect) * np.sin(2 * np.pi * freq * t_seg), gap)) for a in amps])
    meta = {
        "seg_samples": int(seg_dur * fs),
        "gap_samples": int(gap_dur * fs),
        "amps": amps,
        "trim_lead": int(0.03 * fs),
        "trim_tail": int(0.01 * fs),
    }
    return {"signal": tx.astype(np.float32), "meta": meta}


def _lm_analyze_compressor_capture(sig: np.ndarray, meta: Dict[str, Any], fs: int) -> Dict[str, Any]:
    seg_n = meta["seg_samples"]
    gap_n = meta["gap_samples"]
    amps = meta["amps"]
    trim_lead, trim_tail = meta.get("trim_lead", int(0.03 * fs)), meta.get("trim_tail", int(0.01 * fs))

    rms_in_db, rms_out_db = [], []
    for idx, amp in enumerate(amps):
        s0 = idx * (seg_n + gap_n)
        s1 = s0 + seg_n
        seg = sig[s0:s1]
        seg = seg[trim_lead : max(trim_lead, len(seg) - trim_tail)]
        rin = max(amp / np.sqrt(2), 1e-12)
        rout = max(np.sqrt(np.mean(np.square(seg))), 1e-12)
        rms_in_db.append(20 * np.log10(rin))
        rms_out_db.append(20 * np.log10(rout))

    rms_in_db = np.array(rms_in_db)
    rms_out_db = np.array(rms_out_db)
    diff = rms_out_db - rms_in_db
    a_all, _ = np.polyfit(rms_in_db, rms_out_db, 1)
    gain_offset_db = float(np.mean(diff))
    slope_tol, spread_tol = 0.05, 1.0
    no_compression = (abs(a_all - 1.0) < slope_tol) and ((diff.max() - diff.min()) < spread_tol)

    thr, ratio = np.nan, 1.0
    if not no_compression:
        mask = diff < -0.5
        if np.count_nonzero(mask) >= 2:
            x, y = rms_in_db[mask], rms_out_db[mask]
            a, b = np.polyfit(x, y, 1)
            ratio = 1.0 / max(a, 1e-12)
            thr = b / (1 - a) if abs(1 - a) > 1e-6 else np.nan
        else:
            no_compression = True

    return {
        "in_db": rms_in_db,
        "out_db": rms_out_db,
        "diff_db": diff,
        "gain_offset_db": gain_offset_db,
        "no_compression": bool(no_compression),
        "thr_db": float(thr),
        "ratio": float(ratio),
    }

live_measurements.generate_thd_tone = _lm_generate_thd_tone
live_measurements._harmonic_metrics = _lm_harmonic_metrics
live_measurements.analyze_thd_capture = _lm_analyze_thd_capture
live_measurements.generate_compressor_tone = _lm_generate_compressor_tone
live_measurements.analyze_compressor_capture = _lm_analyze_compressor_capture

