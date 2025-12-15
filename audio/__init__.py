"""Single-file audio package bundling device, playrec, and WAV helpers."""
import json
import sys
import types
from typing import List, Optional, Tuple

import numpy as np

# Optional deps
try:  # pragma: no cover
    import sounddevice as sd
    _sd_error: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    sd = None  # type: ignore
    _sd_error = exc

try:  # pragma: no cover
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

import wave


def _create_submodule(name: str) -> types.ModuleType:
    mod = types.ModuleType(__name__ + "." + name)
    sys.modules[mod.__name__] = mod
    setattr(sys.modules[__name__], name, mod)
    return mod


devices = _create_submodule("devices")
playrec = _create_submodule("playrec")
wav_io = _create_submodule("wav_io")

__all__ = ["devices", "playrec", "wav_io"]

# ----------------------------- devices --------------------------------------

def _list_devices(raise_on_error: bool = False) -> Tuple[List[str], List[str]]:
    inputs, outputs = [], []
    if sd is None:
        return inputs, outputs
    try:
        for i, dev in enumerate(sd.query_devices()):
            name = f"{i}: {dev['name']}"
            if dev.get("max_input_channels", 0) > 0:
                inputs.append(name)
            if dev.get("max_output_channels", 0) > 0:
                outputs.append(name)
    except Exception:
        if raise_on_error:
            raise
    return inputs, outputs


def _get_devices_signature() -> Optional[str]:
    if sd is None:
        return None
    try:
        devs = sd.query_devices()
        payload = [
            (i, dev.get("name", ""), dev.get("max_input_channels", 0), dev.get("max_output_channels", 0))
            for i, dev in enumerate(devs)
        ]
        blob = json.dumps(payload, sort_keys=True).encode()
        import hashlib

        return hashlib.sha256(blob).hexdigest()
    except Exception:
        return None


def _parse_device(selection: str):
    try:
        return int(selection.split(":", 1)[0])
    except Exception:
        return None


def _default_samplerate(out_dev):
    if sd is None:
        return 48000
    try:
        dev_info = sd.query_devices(out_dev, "output")
        return int(dev_info["default_samplerate"])
    except Exception:
        try:
            dev_info = sd.query_devices(None, "output")
            return int(dev_info["default_samplerate"])
        except Exception:
            return 48000


devices.list_devices = _list_devices
devices.get_devices_signature = _get_devices_signature
devices.parse_device = _parse_device
devices.default_samplerate = _default_samplerate

# ----------------------------- playrec --------------------------------------

def _play_and_record(
    signal: np.ndarray,
    fs: int,
    in_dev: Optional[int],
    out_dev: Optional[int],
    stop_event,
    log=None,
    input_channels: int = 1,
):
    if sd is None:
        if log:
            log(f"play_and_record unavailable: {_sd_error}")
        return None
    if stop_event.is_set():
        return None
    sd.default.device = (in_dev, out_dev)
    sd.default.samplerate = fs
    channels = int(max(1, input_channels))
    kwargs = {}
    if in_dev is not None or out_dev is not None:
        kwargs["device"] = (in_dev, out_dev)
    try:
        rec = sd.playrec(signal, samplerate=fs, channels=channels, dtype="float32", **kwargs)
        sd.wait()
    except Exception as exc:  # pragma: no cover - runtime capture
        if log:
            log(f"Lỗi playrec: {exc}")
        return None
    if stop_event.is_set():
        return None
    recorded = np.asarray(rec, dtype=np.float32)
    if recorded.ndim == 2 and recorded.shape[1] == 1:
        recorded = recorded[:, 0]
    return recorded


playrec.play_and_record = _play_and_record

# ------------------------------- wav_io -------------------------------------

def _read_wav(path: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
    if sf is not None:
        try:
            data, fs = sf.read(path, always_2d=False)
            data = data.astype(np.float32)
            return fs, data
        except Exception:
            pass
    try:
        with wave.open(path, "rb") as wf:
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


def _write_wav(path: str, data: np.ndarray, fs: int) -> bool:
    if sf is not None:
        try:
            sf.write(path, data, fs)
            return True
        except Exception:
            pass
    try:
        scaled = np.clip(data, -1.0, 1.0)
        scaled = (scaled * 32767).astype("<i2")
        with wave.open(path, "wb") as wf:
            channels = 1 if scaled.ndim == 1 else scaled.shape[1]
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(scaled.tobytes())
        return True
    except Exception:
        return False


wav_io.read_wav = _read_wav
wav_io.write_wav = _write_wav

