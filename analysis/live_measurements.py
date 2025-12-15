import csv
import os
import time
from typing import Dict, Any, Tuple

import numpy as np

from analysis import thd
from audio import wav_io

BASE_DURATION = 2.0
FS_DEFAULT = 48000


def _ensure_out_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, "out", "live")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_artifacts(tag: str, tx: np.ndarray, rx: np.ndarray, fs: int, base_dir: str) -> Dict[str, str]:
    out_dir = _ensure_out_dir(base_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tx_path = os.path.join(out_dir, f"{tag}_{stamp}_tx.wav")
    rx_path = os.path.join(out_dir, f"{tag}_{stamp}_rx.wav")
    wav_io.write_wav(tx_path, tx, fs)
    wav_io.write_wav(rx_path, rx, fs)
    return {"tx": tx_path, "rx": rx_path}


def append_csv_row(row: Tuple[str, ...], base_dir: str, filename: str = "ket_qua_do.csv") -> str:
    out_dir = _ensure_out_dir(base_dir)
    path = os.path.join(out_dir, filename)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    return path


def generate_thd_tone(freq: float, amp: float, fs: int, duration: float = BASE_DURATION) -> np.ndarray:
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sine = amp * np.sin(2 * np.pi * freq * t)
    return np.column_stack((sine, np.zeros_like(sine))).astype(np.float32)


def _harmonic_metrics(signal: np.ndarray, fs: int, freq: float, max_h: int) -> Tuple[Dict[int, float], float, float]:
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


def analyze_thd_capture(recorded: np.ndarray, fs: int, freq: float, hmax: int) -> Dict[str, Any]:
    trimmed = np.asarray(recorded, dtype=np.float32).flatten()
    trimmed = trimmed[int(0.05 * fs) :]
    peak = float(np.max(np.abs(trimmed)) + 1e-12)
    normalized = trimmed / peak
    harmonics, thd_percent, thd_db = _harmonic_metrics(normalized, fs, freq, hmax)
    thd_metrics = thd.compute_thd(normalized, fs, freq, hmax)
    thd_metrics.update({
        "harmonics_manual": harmonics,
        "thd_percent_manual": thd_percent,
        "thd_db_manual": thd_db,
        "normalized_signal": normalized,
    })
    return thd_metrics


def generate_compressor_tone(freq: float, fs: int, amp_max: float = 1.36) -> Dict[str, Any]:
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


def analyze_compressor_capture(sig: np.ndarray, meta: Dict[str, Any], fs: int) -> Dict[str, Any]:
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
    a_all, b_all = np.polyfit(rms_in_db, rms_out_db, 1)
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
