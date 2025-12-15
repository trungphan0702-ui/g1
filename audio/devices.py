import json
import hashlib
from typing import List, Optional, Tuple

try:  # Optional dependency: sounddevice may be absent in offline environments
    import sounddevice as sd
    _sd_error: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - best-effort fallback
    sd = None  # type: ignore
    _sd_error = exc


def list_devices(raise_on_error: bool = False) -> Tuple[List[str], List[str]]:
    """Return (inputs, outputs) as display strings.

    When ``raise_on_error`` is True, propagate the sounddevice exception so the
    caller can log it; otherwise silently returns empty lists on failure.
    """

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


def get_devices_signature() -> Optional[str]:
    """Return a stable hash of the current device list for change detection."""

    if sd is None:
        return None
    try:
        devs = sd.query_devices()
        payload = [
            (i, dev.get("name", ""), dev.get("max_input_channels", 0), dev.get("max_output_channels", 0))
            for i, dev in enumerate(devs)
        ]
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()
    except Exception:
        return None


def parse_device(selection: str):
    try:
        return int(selection.split(":", 1)[0])
    except Exception:
        return None


def default_samplerate(out_dev):
    if sd is None:
        return 48000
    try:
        dev_info = sd.query_devices(out_dev, 'output')
        return int(dev_info['default_samplerate'])
    except Exception:
        try:
            dev_info = sd.query_devices(None, 'output')
            return int(dev_info['default_samplerate'])
        except Exception:
            return 48000
