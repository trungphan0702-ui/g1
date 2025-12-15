import threading
from typing import Callable, Optional


def run_in_thread(target: Callable, stop_event: threading.Event, name: Optional[str] = None) -> threading.Thread:
    """Launch a target in a background daemon thread."""
    thread = threading.Thread(target=target, name=name or target.__name__)
    thread.daemon = True
    stop_event.clear()
    thread.start()
    return thread
