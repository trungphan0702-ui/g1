import numpy as np
from typing import Dict, Any


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
    if sig.ndim > 1:
        sig = sig[:, 0]
    # Attack uses shorter time-constant than release to mimic envelope follower
    attack_ms = win_ms
    release_ms = max(win_ms * 10.0, win_ms)
    attack_coeff = float(np.exp(-1.0 / (attack_ms / 1000.0 * fs)))
    release_coeff = float(np.exp(-1.0 / (release_ms / 1000.0 * fs)))

    env = np.zeros_like(sig, dtype=np.float32)
    last = 0.0
    for i, sample in enumerate(np.abs(sig)):
        coeff = attack_coeff if sample > last else release_coeff
        last = coeff * last + (1.0 - coeff) * sample
        env[i] = last

    if len(env) < 10 or not np.isfinite(np.max(env)):
        return {'attack_ms': float('nan'), 'release_ms': float('nan')}

    # Identify rising and falling edges
    diff = np.diff(env)
    attack_start = int(np.argmax(diff))
    release_start = int(np.argmin(diff)) if np.any(diff < 0) else len(env) // 2

    target = float(np.max(env))
    attack_level = target * 0.9
    release_level = target * 0.37

    attack_slice = env[attack_start:]
    attack_offset = int(np.argmax(attack_slice >= attack_level)) if attack_slice.size else 0
    attack_idx = attack_start + attack_offset

    release_slice = env[release_start:]
    if release_slice.size == 0:
        release_idx = len(env) - 1
    else:
        release_offset = int(np.argmax(release_slice <= release_level))
        if release_slice[release_offset] > release_level and release_offset == 0:
            release_offset = len(release_slice) - 1
        release_idx = release_start + release_offset

    if release_idx <= attack_idx:
        release_idx = min(len(env) - 1, attack_idx + 1)

    return {
        'attack_ms': attack_idx / fs * 1000.0,
        'release_ms': release_idx / fs * 1000.0,
    }


def compare_attack_release(input_sig: np.ndarray, output_sig: np.ndarray, fs: int, win_ms: float) -> Dict[str, Any]:
    in_times = attack_release_times(input_sig, fs, win_ms)
    out_times = attack_release_times(output_sig, fs, win_ms)
    return {
        'input': in_times,
        'output': out_times,
        'delta_attack': out_times['attack_ms'] - in_times['attack_ms'],
        'delta_release': out_times['release_ms'] - in_times['release_ms'],
    }
