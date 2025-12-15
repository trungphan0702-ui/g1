import threading
from typing import Callable

class UILogger:
    """Thread-safe logger that forwards messages to a UI callback."""

    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self._lock = threading.Lock()

    def log(self, message: str) -> None:
        with self._lock:
            if self.callback:
                self.callback(message)

    def banner(self, message: str) -> None:
        self.log(f"[==] {message}")

    def info(self, message: str) -> None:
        self.log(f"[INFO] {message}")

    def warn(self, message: str) -> None:
        self.log(f"[WARN] {message}")

    def error(self, message: str) -> None:
        self.log(f"[ERROR] {message}")
