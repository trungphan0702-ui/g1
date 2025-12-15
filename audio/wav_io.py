import numpy as np
import wave
from typing import Tuple, Optional

import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None


def read_wav(path: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
    if sf is not None:
        try:
            data, fs = sf.read(path, always_2d=False)
            data = data.astype(np.float32)
            return fs, data
        except Exception:
            pass
    try:
        with wave.open(path, 'rb') as wf:
            fs = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            channels = wf.getnchannels()
            if channels > 1:
                data = data.reshape(-1, channels)
            return fs, data
    except Exception:
        return None, None
    return None, None


def write_wav(path: str, data: np.ndarray, fs: int) -> bool:
    if sf is not None:
        try:
            sf.write(path, data, fs)
            return True
        except Exception:
            pass
    try:
        scaled = np.clip(data, -1.0, 1.0)
        scaled = (scaled * 32767).astype('<i2')
        with wave.open(path, 'wb') as wf:
            if scaled.ndim == 1:
                channels = 1
            else:
                channels = scaled.shape[1]
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(scaled.tobytes())
        return True
    except Exception:
        return False
