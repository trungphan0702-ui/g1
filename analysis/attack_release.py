import os
import numpy as np
from typing import Dict, Any, Optional


def generate_step_tone(freq: float, fs: int, amp: float = 0.7, duration: float = 2.0) -> np.ndarray:
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = amp * np.sin(2 * np.pi * freq * t)
    # amplitude steps: low -> high -> low to expose attack and release
    env = np.ones_like(tone) * 0.3
    attack_idx = len(env) // 4
    release_idx = 3 * len(env) // 4
    env[attack_idx:release_idx] = 1.0
    env[release_idx:] = 0.3
    return (tone * env).astype(np.float32)


def envelope_rms(sig: np.ndarray, fs: int, win_ms: float) -> np.ndarray:
    win = int(max(1, fs * win_ms / 1000))
    if sig.ndim > 1:
        sig = sig[:, 0]
    padded = np.pad(sig ** 2, (win, win))
    cumsum = np.cumsum(padded)
    rms = np.sqrt((cumsum[2 * win:] - cumsum[:-2 * win]) / max(2 * win, 1))
    return rms


def attack_release_times(sig: np.ndarray, fs: int, win_ms: float) -> Dict[str, float]:
    """Estimate attack/release using RMS envelope crossings.

    The previous version used an envelope follower with a long implicit time
    constant, which overstated the timing on synthetic step tones. This version
    uses a short RMS window (``win_ms``) and 10→90% / 90→10% crossings around
    the rising and falling sections of the generated step tone.
    """

    if sig.ndim > 1:
        sig = sig[:, 0]

    env = envelope_rms(sig, fs, win_ms)
    if len(env) < 10 or not np.isfinite(np.max(env)):
        return {'attack_ms': float('nan'), 'release_ms': float('nan')}

    n = len(env)
    q1, q3 = n // 4, 3 * n // 4
    low_level = float(np.median(env[:q1]))
    high_level = float(np.median(env[q1:q3]))
    tail_level = float(np.median(env[q3:]))

    # Protect against degenerate signals
    peak = float(np.max(env))
    if peak < 1e-12:
        return {'attack_ms': float('nan'), 'release_ms': float('nan')}

    atk_start_lvl = low_level + 0.1 * (high_level - low_level)
    atk_end_lvl = low_level + 0.9 * (high_level - low_level)
    rel_start_lvl = high_level - 0.1 * (high_level - tail_level)
    rel_end_lvl = high_level - 0.9 * (high_level - tail_level)

    search_rise = env[q1 - n // 10 : q3]
    search_fall = env[q3 - n // 10 :]

    def _crossing(x: np.ndarray, level: float, direction: str = 'up') -> Optional[int]:
        if direction == 'up':
            idxs = np.nonzero(x >= level)[0]
        else:
            idxs = np.nonzero(x <= level)[0]
        return int(idxs[0]) if idxs.size else None

    atk_start_rel = _crossing(search_rise, atk_start_lvl, 'up')
    atk_end_rel = _crossing(search_rise, atk_end_lvl, 'up')
    rel_start_rel = _crossing(search_fall, rel_start_lvl, 'down')
    rel_end_rel = _crossing(search_fall, rel_end_lvl, 'down')

    attack_idx = atk_end_idx = release_idx = None
    if atk_start_rel is not None and atk_end_rel is not None:
        attack_idx = (q1 - n // 10) + atk_start_rel
        atk_end_idx = (q1 - n // 10) + atk_end_rel
    if rel_start_rel is not None and rel_end_rel is not None:
        release_idx = (q3 - n // 10) + rel_end_rel
        rel_start_idx = (q3 - n // 10) + rel_start_rel
    else:
        rel_start_idx = None

    if atk_end_idx is None or attack_idx is None:
        attack_ms = float('nan')
    else:
        attack_ms = (atk_end_idx - attack_idx) / fs * 1000.0

    if release_idx is None or rel_start_idx is None:
        release_ms = float('nan')
    else:
        release_ms = (release_idx - rel_start_idx) / fs * 1000.0

    if os.getenv("DSP_DEBUG"):
        print(
            f"[DSP_DEBUG][AR] low={low_level:.3e}, high={high_level:.3e}, tail={tail_level:.3e}, "
            f"atk_idx={attack_idx}, atk_end={atk_end_idx}, rel_start={rel_start_idx}, rel_end={release_idx}"
        )

    return {'attack_ms': attack_ms, 'release_ms': release_ms}


def compare_attack_release(input_sig: np.ndarray, output_sig: np.ndarray, fs: int, win_ms: float) -> Dict[str, Any]:
    in_times = attack_release_times(input_sig, fs, win_ms)
    out_times = attack_release_times(output_sig, fs, win_ms)
    return {
        'input': in_times,
        'output': out_times,
        'delta_attack': out_times['attack_ms'] - in_times['attack_ms'],
        'delta_release': out_times['release_ms'] - in_times['release_ms'],
    }
