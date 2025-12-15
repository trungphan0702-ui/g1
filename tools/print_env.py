"""Print environment and audio device info for troubleshooting."""

import json
import sys


def main() -> int:
    print("Python:", sys.version.replace("\n", " "))

    try:
        import numpy as np  # type: ignore
        print("numpy:", np.__version__)
    except Exception as exc:  # pragma: no cover - diagnostic helper
        print(f"numpy: MISSING ({exc})")

    try:
        import sounddevice as sd  # type: ignore

        print("sounddevice:", sd.__version__)
        try:
            devices = sd.query_devices()
            summarized = [
                {
                    "id": idx,
                    "name": d.get("name"),
                    "max_input_channels": d.get("max_input_channels", 0),
                    "max_output_channels": d.get("max_output_channels", 0),
                    "default_samplerate": d.get("default_samplerate"),
                }
                for idx, d in enumerate(devices)
            ]
            print("devices:")
            print(json.dumps(summarized, indent=2))
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            print(f"devices: unable to query ({exc})")
    except Exception as exc:  # pragma: no cover - diagnostic helper
        print(f"sounddevice: MISSING ({exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
