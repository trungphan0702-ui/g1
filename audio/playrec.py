import numpy as np
from typing import Optional

try:
    import sounddevice as sd
    _sd_error = None
except Exception as exc:  # ImportError or PortAudio missing
    sd = None
    _sd_error = exc


def play_and_record(signal: np.ndarray, fs: int, in_dev: Optional[int], out_dev: Optional[int], stop_event, log=None):
    if sd is None:
        if log:
            log(f"play_and_record unavailable: { _sd_error }")
        return None
    if stop_event.is_set():
        return None
    sd.default.device = (in_dev, out_dev)
    sd.default.samplerate = fs
    channels = 1 if signal.ndim == 1 else signal.shape[1]
    try:
        rec = sd.playrec(signal, samplerate=fs, channels=channels, dtype='float32')
        sd.wait()
    except Exception as exc:
        if log:
            log(f"Lỗi playrec: {exc}")
        return None
    if stop_event.is_set():
        return None
    recorded = np.asarray(rec, dtype=np.float32)
    if recorded.ndim == 2 and recorded.shape[1] == 1:
        recorded = recorded[:, 0]
    return recorded
