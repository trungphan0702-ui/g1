import argparse
import csv
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, os.pardir)))

from analysis import attack_release, compare, compressor, thd
from audio import wav_io


@dataclass
class BenchConfig:
    fs: int = 48000
    freq: float = 1000.0
    amp: float = 0.7
    duration: float = 2.0
    thd_max_h: int = 5
    thd_fund_band_bins: int = 2
    thd_window: str = "hann"
    thd_nfft: Optional[int] = None
    attack_rms_win_ms: float = 5.0
    compressor_amp_max: float = 1.0
    compressor_threshold_db: float = -12.0
    compressor_ratio: float = 4.0
    compressor_makeup_db: float = 0.0
    compressor_knee_db: float = 0.0
    compressor_attack_ms: float = 10.0
    compressor_release_ms: float = 100.0

    def update(self, data: Dict[str, Any]) -> None:
        for key, val in data.items():
            if hasattr(self, key):
                setattr(self, key, val)


@dataclass
class BenchCase:
    name: str
    category: str
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str, verbose: bool = False) -> None:
    if verbose or os.getenv("DSP_DEBUG"):
        print(msg)


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sine_with_harmonic(freq: float, fs: int, duration: float, amp: float, harmonic: Optional[Dict[str, Any]] = None) -> np.ndarray:
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = amp * np.sin(2 * np.pi * freq * t)
    if harmonic:
        order = int(harmonic.get("order", 2))
        level_db = float(harmonic.get("level_db", -30.0))
        harm_amp = amp * 10 ** (level_db / 20.0)
        sig += harm_amp * np.sin(2 * np.pi * freq * order * t)
    return sig.astype(np.float32)


def _flatten_metrics(case: BenchCase) -> Dict[str, Any]:
    row: Dict[str, Any] = {"name": case.name, "category": case.category}
    for key, val in case.metrics.items():
        if isinstance(val, (list, tuple, np.ndarray)):
            row[key] = json.dumps(val if not isinstance(val, np.ndarray) else val.tolist())
        elif isinstance(val, dict):
            row[key] = json.dumps(val)
        else:
            row[key] = val
    if case.notes:
        row["notes"] = " | ".join(case.notes)
    if case.params:
        row["params"] = json.dumps(case.params)
    return row


def _collect_env() -> Dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Bench runners
# ---------------------------------------------------------------------------

def run_thd_cases(cfg: BenchConfig, cases_cfg: List[Dict[str, Any]], verbose: bool) -> List[BenchCase]:
    results: List[BenchCase] = []
    for entry in cases_cfg:
        name = entry.get("name", "thd_case")
        params = {
            "freq": entry.get("freq", cfg.freq),
            "amp": entry.get("amp", cfg.amp),
            "duration": entry.get("duration", cfg.duration),
            "max_h": entry.get("max_h", cfg.thd_max_h),
            "window": entry.get("window", cfg.thd_window),
            "fundamental_band_bins": entry.get("fundamental_band_bins", cfg.thd_fund_band_bins),
            "nfft": entry.get("nfft", cfg.thd_nfft),
        }

        if entry.get("type") == "file":
            wav_path = entry.get("input_wav") or ""
            if not wav_path or not os.path.isfile(wav_path):
                results.append(BenchCase(name=name, category="thd", params=params, notes=["Skipped: missing input_wav."]))
                continue
            fs, data = wav_io.read_wav(wav_path)
            if fs is None:
                results.append(BenchCase(name=name, category="thd", params=params, notes=["Skipped: cannot read WAV."]))
                continue
            signal = data
        else:
            signal = _sine_with_harmonic(
                params["freq"], cfg.fs, params["duration"], params["amp"], harmonic=entry.get("harmonic")
            )
            fs = cfg.fs

        _log(f"[THD] Running {name} (fs={fs}, freq={params['freq']})", verbose)
        metrics = thd.compute_thd(
            signal,
            fs,
            params["freq"],
            max_h=params["max_h"],
            window=params["window"],
            fundamental_band_bins=params["fundamental_band_bins"],
            nfft=params["nfft"],
        )
        notes: List[str] = []
        if entry.get("expected"):
            exp = entry["expected"]
            if "thdn_db_max" in exp and metrics["thdn_db"] > exp["thdn_db_max"]:
                notes.append(f"Sanity: THD+N {metrics['thdn_db']:.2f} dB exceeds expected max {exp['thdn_db_max']}")
            if "thd_db_min" in exp and metrics["thd_db"] < exp["thd_db_min"]:
                notes.append(f"Sanity: THD {metrics['thd_db']:.2f} dB below expected min {exp['thd_db_min']}")
        results.append(BenchCase(name=name, category="thd", params=params, metrics=metrics, notes=notes))
    return results


def run_attack_release_cases(cfg: BenchConfig, cases_cfg: List[Dict[str, Any]], verbose: bool) -> List[BenchCase]:
    results: List[BenchCase] = []
    for entry in cases_cfg:
        name = entry.get("name", "attack_release")
        freq = entry.get("freq", cfg.freq)
        amp = entry.get("amp", cfg.amp)
        win_ms = entry.get("rms_win_ms", cfg.attack_rms_win_ms)
        duration = entry.get("duration", cfg.duration)

        tone = attack_release.generate_step_tone(freq, cfg.fs, amp=amp, duration=duration)
        metrics = attack_release.attack_release_times(tone, cfg.fs, win_ms)
        params = {"freq": freq, "amp": amp, "rms_win_ms": win_ms, "fs": cfg.fs, "duration": duration}
        results.append(BenchCase(name=name, category="attack_release", params=params, metrics=metrics))
        _log(f"[AR] {name}: attack={metrics['attack_ms']:.1f} ms, release={metrics['release_ms']:.1f} ms", verbose)
    return results


def run_compressor_cases(cfg: BenchConfig, cases_cfg: List[Dict[str, Any]], verbose: bool) -> List[BenchCase]:
    results: List[BenchCase] = []
    for entry in cases_cfg:
        name = entry.get("name", "compressor")
        freq = entry.get("freq", cfg.freq)
        amp_max = entry.get("amp_max", cfg.compressor_amp_max)
        apply = bool(entry.get("apply_compressor", False))
        comp_params = {
            "threshold_db": entry.get("threshold_db", cfg.compressor_threshold_db),
            "ratio": entry.get("ratio", cfg.compressor_ratio),
            "makeup_db": entry.get("makeup_db", cfg.compressor_makeup_db),
            "knee_db": entry.get("knee_db", cfg.compressor_knee_db),
            "attack_ms": entry.get("attack_ms", cfg.compressor_attack_ms),
            "release_ms": entry.get("release_ms", cfg.compressor_release_ms),
        }

        tone_meta = compressor.build_stepped_tone(freq, cfg.fs, amp_max=amp_max)
        tx = tone_meta["signal"]
        rx = tx
        if apply:
            rx = compressor.apply_compressor(
                tx,
                threshold_db=comp_params["threshold_db"],
                ratio=comp_params["ratio"],
                makeup_db=comp_params["makeup_db"],
                knee_db=comp_params["knee_db"],
                attack_ms=comp_params["attack_ms"],
                release_ms=comp_params["release_ms"],
                fs=cfg.fs,
            )

        metrics = compressor.compression_curve(rx, tone_meta["meta"], cfg.fs, freq)
        params = {"freq": freq, "amp_max": amp_max, **comp_params, "applied": apply, "fs": cfg.fs}
        results.append(BenchCase(name=name, category="compressor", params=params, metrics=metrics))
        _log(
            f"[COMP] {name}: thr={metrics['thr_db']}, ratio={metrics['ratio']}, gain_off={metrics['gain_offset_db']:+.2f} dB",
            verbose,
        )
    return results


def run_compare_cases(cfg: BenchConfig, cases_cfg: List[Dict[str, Any]], verbose: bool) -> List[BenchCase]:
    results: List[BenchCase] = []
    for entry in cases_cfg:
        name = entry.get("name", "compare")
        inp = entry.get("input_wav") or ""
        out = entry.get("output_wav") or ""
        freq = entry.get("freq", cfg.freq)
        hmax = entry.get("hmax", cfg.thd_max_h)
        if not (inp and out and os.path.isfile(inp) and os.path.isfile(out)):
            results.append(BenchCase(name=name, category="compare", params={"input": inp, "output": out}, notes=["Skipped: missing input/output wav."]))
            continue

        fs_in, sig_in = wav_io.read_wav(inp)
        fs_out, sig_out = wav_io.read_wav(out)
        if fs_in is None or fs_out is None or fs_in != fs_out:
            results.append(BenchCase(name=name, category="compare", params={"input": inp, "output": out}, notes=["Skipped: fs mismatch or read error."]))
            continue

        aligned_ref, aligned_tgt, lag = compare.align_signals(sig_in, sig_out)
        gain_matched, gain_error_db = compare.gain_match(aligned_ref, aligned_tgt)
        metrics = compare.residual_metrics(aligned_ref, gain_matched, fs_in, freq, hmax)
        metrics.update({"latency_samples": lag, "latency_ms": lag / fs_in * 1000.0, "gain_error_db": gain_error_db})
        params = {"freq": freq, "fs": fs_in, "hmax": hmax, "input_wav": inp, "output_wav": out}
        results.append(BenchCase(name=name, category="compare", params=params, metrics=metrics))
        _log(
            f"[CMP] {name}: latency={metrics['latency_ms']:.2f} ms, gain_err={metrics['gain_error_db']:+.2f} dB, SNR={metrics['snr_db']:.2f} dB",
            verbose,
        )
    return results


# ---------------------------------------------------------------------------
# CSV / JSON helpers
# ---------------------------------------------------------------------------

def write_outputs(cases: List[BenchCase], json_path: str, csv_path: str) -> None:
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    serializable_cases = []
    for case in cases:
        serializable_cases.append(
            {
                "name": case.name,
                "category": case.category,
                "params": _to_jsonable(case.params),
                "metrics": _to_jsonable(case.metrics),
                "notes": case.notes,
            }
        )

    data = {"env": _collect_env(), "cases": serializable_cases}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    all_rows = [_flatten_metrics(c) for c in cases]
    fieldnames: List[str] = []
    for row in all_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline DSP benchmark harness")
    parser.add_argument("--config", default="tools/bench_config.json", help="Path to bench config JSON")
    parser.add_argument("--out", default="out/bench_results.json", help="Path to JSON results")
    parser.add_argument("--csv", default="out/bench_results.csv", help="Path to CSV results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    cfg_data = _load_config(args.config) if os.path.isfile(args.config) else {}
    cfg = BenchConfig()
    cfg.update(cfg_data.get("defaults", {}))

    thd_cases_cfg = cfg_data.get("thd_cases", [])
    ar_cases_cfg = cfg_data.get("attack_release_cases", [])
    comp_cases_cfg = cfg_data.get("compressor_cases", [])
    cmp_cases_cfg = cfg_data.get("compare_cases", [])

    cases: List[BenchCase] = []
    cases.extend(run_thd_cases(cfg, thd_cases_cfg, args.verbose))
    cases.extend(run_attack_release_cases(cfg, ar_cases_cfg, args.verbose))
    cases.extend(run_compressor_cases(cfg, comp_cases_cfg, args.verbose))
    cases.extend(run_compare_cases(cfg, cmp_cases_cfg, args.verbose))

    write_outputs(cases, args.out, args.csv)
    print(f"Wrote results to {args.out} and {args.csv}")

    # Console summary
    for case in cases:
        msg = f"- {case.name} [{case.category}]"
        if case.notes:
            msg += " | " + "; ".join(case.notes)
        elif case.metrics:
            if case.category == "thd":
                msg += f" | THD {case.metrics.get('thd_db', np.nan):.2f} dB, THD+N {case.metrics.get('thdn_db', np.nan):.2f} dB"
            elif case.category == "attack_release":
                msg += f" | Attack {case.metrics.get('attack_ms', np.nan):.1f} ms / Release {case.metrics.get('release_ms', np.nan):.1f} ms"
            elif case.category == "compressor":
                msg += f" | Thr {case.metrics.get('thr_db')} dB, Ratio {case.metrics.get('ratio')}"
            elif case.category == "compare":
                msg += f" | Latency {case.metrics.get('latency_ms', np.nan):.2f} ms, Gain {case.metrics.get('gain_error_db', np.nan):+.2f} dB"
        print(msg)


if __name__ == "__main__":
    main()
