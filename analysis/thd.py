import os
import numpy as np
from typing import Dict, Any, Optional


def normalize_thd_result(data: Dict[str, Any], fallback_db: float = 0.0) -> Dict[str, Any]:
    """Normalize THD/THD+N result keys and guarantee required fields.

    Accepted aliases for THD+N include "thdn", "thdn_dB", "thd+n_db". The
    returned dictionary is always safe to index with ``thd_db`` and
    ``thdn_db``. When a key cannot be computed, a numeric fallback is used to
    avoid runtime crashes while still making the metric obviously poor.
    """

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


def compute_thd(
    signal: np.ndarray,
    fs: int,
    freq: float,
    max_h: int = 5,
    window: str = "hann",
    fundamental_band_bins: int = 2,
    nfft: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute THD/THD+N for ``signal`` using a simple FFT method.

    Parameters mirror the GUI entry points to keep the API stable while
    exposing extra knobs (window, fundamental band width, optional zero-pad
    length) for offline benchmarking.
    """

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
    # Avoid DC contamination
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
        'fundamental_mag': fund_mag,
        'harmonics_dbc': harmonics,
        'thd_percent': thd_percent,
        'thd_ratio': thd_ratio,
        'thd_db': thd_db,
        'thdn_ratio': thdn_ratio,
        'thdn_db': thdn_db,
        'freqs': freqs,
        'spectrum': 20 * np.log10(mag + 1e-12),
        'fs': fs,
        'fund_freq': freq,
        'nfft': nfft_use,
        'window': 'hann' if window == 'hanning' else window,
        'fund_band_bins': band_bins,
    }

    normalized = normalize_thd_result(result)

    if os.getenv("DSP_DEBUG"):
        print(
            f"[DSP_DEBUG] compute_thd: fund_idx={fund_idx}, fund_mag={fund_mag:.3e}, "
            f"thd_db={normalized['thd_db']:.2f}, thdn_db={normalized['thdn_db']:.2f}"
        )

    return normalized
