"""Single-file utils package bundling logging, plot_windows, and threading helpers."""
import sys
import types
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _create_submodule(name: str) -> types.ModuleType:
    mod = types.ModuleType(__name__ + "." + name)
    sys.modules[mod.__name__] = mod
    setattr(sys.modules[__name__], name, mod)
    return mod


logging = _create_submodule("logging")
plot_windows = _create_submodule("plot_windows")
threading = _create_submodule("threading")

__all__ = ["logging", "plot_windows", "threading"]

# ----------------------------- logging --------------------------------------
import threading as _threading


class _UILogger:
    """Thread-safe logger that forwards messages to a UI callback."""

    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self._lock = _threading.Lock()

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


logging.UILogger = _UILogger

# --------------------------- plot_windows -----------------------------------
import datetime
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class _PlotWindowManager:
    """Manage snapshot-style matplotlib windows embedded in Tk Toplevels."""

    def __init__(self, root: tk.Tk, max_windows: int = 10, log: Optional[Callable[[str], None]] = None):
        self.root = root
        self.max_windows = max_windows
        self.log = log
        self.windows: List[Tuple[tk.Toplevel, Figure]] = []

    def _log(self, msg: str):
        if self.log:
            self.log(msg)

    def _close_window(self, top: tk.Toplevel, fig: Figure):
        try:
            if fig:
                fig.clf()
            top.destroy()
        except Exception:
            pass
        self.windows = [(t, f) for (t, f) in self.windows if t != top]

    def _register(self, top: tk.Toplevel, fig: Figure):
        self.windows.append((top, fig))
        top.protocol("WM_DELETE_WINDOW", lambda t=top, f=fig: self._close_window(t, f))

    def _maybe_trim_windows(self):
        if len(self.windows) >= self.max_windows:
            oldest_top, oldest_fig = self.windows.pop(0)
            self._log("Đã đóng snapshot cũ để giải phóng bộ nhớ (tối đa %d cửa sổ)." % self.max_windows)
            self._close_window(oldest_top, oldest_fig)

    def _create_window(self, title: str) -> Tuple[tk.Toplevel, Figure, FigureCanvasTkAgg, ttk.Frame]:
        self._maybe_trim_windows()
        top = tk.Toplevel(self.root)
        top.title(title)
        fig = Figure(figsize=(9, 6))
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        controls = ttk.Frame(top)
        controls.pack(fill="x")
        btn = ttk.Button(controls, text="Save PNG", command=lambda f=fig: self._save_png(f))
        btn.pack(side="right", padx=6, pady=4)

        self._register(top, fig)
        return top, fig, canvas, controls

    def _save_png(self, fig: Figure):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"Đã lưu hình: {path}")

    def open_thd_snapshot(
        self,
        signal: np.ndarray,
        fs: int,
        metrics: Dict[str, float],
        freq: float,
        hmax: int,
    ):
        title = f"THD Plot – {datetime.datetime.now().strftime('%H:%M:%S')}"
        _, fig, canvas, controls = self._create_window(title)
        ax_wave = fig.add_subplot(2, 1, 1)
        ax_fft = fig.add_subplot(2, 1, 2)

        sig = np.asarray(signal, dtype=np.float32).flatten()
        n_show = min(len(sig), int(fs * 0.05))
        t = np.arange(n_show) / fs
        ax_wave.plot(t * 1000.0, sig[:n_show])
        ax_wave.set_title("Waveform (50 ms snippet)")
        ax_wave.set_xlabel("Time (ms)")
        ax_wave.set_ylabel("Amplitude")

        freqs = metrics.get("freqs")
        spectrum = metrics.get("spectrum")
        if freqs is not None and spectrum is not None:
            ax_fft.plot(freqs, spectrum, label="Magnitude (dB)")
            ax_fft.set_xlim(0, fs / 2)
            ax_fft.set_xlabel("Frequency (Hz)")
            ax_fft.set_ylabel("dBFS")
            for h in range(1, hmax + 1):
                hf = h * freq
                idx = int(np.argmin(np.abs(freqs - hf)))
                if idx < len(freqs):
                    ax_fft.axvline(freqs[idx], color="red", linestyle="--", alpha=0.5)
                    ax_fft.text(freqs[idx], spectrum[idx], f"H{h}", rotation=90, va="bottom", ha="center", fontsize=8)
            ax_fft.legend()

        thd_db = metrics.get("thd_db", float("nan"))
        thdn_db = metrics.get("thdn_db", float("nan"))
        ax_fft.set_title(f"THD {thd_db:.2f} dB | THD+N {thdn_db:.2f} dB")

        def _export_csv():
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
            if not path or freqs is None or spectrum is None:
                return
            import csv

            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["freq_hz", "spectrum_db"])
                for f_hz, mag_db in zip(freqs, spectrum):
                    w.writerow([float(f_hz), float(mag_db)])
            self._log(f"Đã xuất CSV: {path}")

        ttk.Button(controls, text="Export CSV", command=_export_csv).pack(side="right", padx=6, pady=4)

        fig.tight_layout()
        canvas.draw()

    def open_compression_snapshot(self, curves: Sequence[Tuple[str, Dict[str, np.ndarray]]]):
        title = f"Compression Curve – {datetime.datetime.now().strftime('%H:%M:%S')}"
        _, fig, canvas, controls = self._create_window(title)
        ax = fig.add_subplot(1, 1, 1)

        for label, curve in curves:
            x = curve.get("in_db")
            y = curve.get("out_db")
            thr = curve.get("thr_db")
            ratio = curve.get("ratio")
            gain_offset = curve.get("gain_offset_db")
            no_comp = curve.get("no_compression")
            if x is None or y is None:
                continue
            ax.plot(x, y, ".-", label=label)
            ax.plot(x, x, "k--", linewidth=1, alpha=0.6)
            if no_comp:
                xgrid = np.linspace(min(x), max(x), 100)
                ax.plot(xgrid, xgrid + (gain_offset or 0.0), "g:", label=f"{label} 1:1 offset")
            elif thr is not None and np.isfinite(thr):
                ax.axvline(thr, color="g", linestyle="--", alpha=0.7)
                ax.text(thr, min(y), f"Thr {thr:.2f}", rotation=90, va="bottom", ha="center", fontsize=8)
            if ratio is not None and ratio > 0:
                ax.text(x[-1], y[-1], f"{ratio:.2f}:1", fontsize=8)

        ax.set_xlim(-40, 0)
        ax.set_ylim(-40, 0)
        ax.set_xlabel("Input RMS (dBFS)")
        ax.set_ylabel("Output RMS (dBFS)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        subtitle = []
        for label, curve in curves:
            thr = curve.get("thr_db")
            ratio = curve.get("ratio")
            makeup = curve.get("gain_offset_db")
            nc = curve.get("no_compression")
            if nc:
                subtitle.append(f"{label}: passthrough (gain {makeup:+.2f} dB)")
            else:
                subtitle.append(f"{label}: Thr {thr:.2f} dB, Ratio {ratio:.2f}, Gain {makeup:+.2f} dB")
        ax.set_title(" | ".join(subtitle))

        def _export_csv():
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
            if not path:
                return
            import csv

            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                headers = ["label", "input_db", "output_db"]
                w.writerow(headers)
                for label, curve in curves:
                    x = curve.get("in_db")
                    y = curve.get("out_db")
                    if x is None or y is None:
                        continue
                    for xi, yi in zip(x, y):
                        w.writerow([label, float(xi), float(yi)])
            self._log(f"Đã xuất CSV: {path}")

        ttk.Button(controls, text="Export CSV", command=_export_csv).pack(side="right", padx=6, pady=4)

        fig.tight_layout()
        canvas.draw()

    def open_ar_snapshot(self, signal: np.ndarray, fs: int, rms_win: float, metrics: Dict[str, float]):
        title = f"Attack/Release – {datetime.datetime.now().strftime('%H:%M:%S')}"
        _, fig, canvas, controls = self._create_window(title)
        ax_wave = fig.add_subplot(2, 1, 1)
        ax_env = fig.add_subplot(2, 1, 2)

        sig = np.asarray(signal, dtype=np.float32).flatten()
        t = np.arange(len(sig)) / fs
        ax_wave.plot(t, sig)
        ax_wave.set_title("Waveform")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")

        from analysis import attack_release  # type: ignore

        env = attack_release.envelope_rms(sig, fs, rms_win)
        env_t = np.arange(len(env)) / fs
        ax_env.plot(env_t, env, label="RMS envelope")

        def _find_markers(env_arr: np.ndarray):
            n = len(env_arr)
            q1, q3 = n // 4, 3 * n // 4
            low = float(np.median(env_arr[:q1]))
            high = float(np.median(env_arr[q1:q3]))
            tail = float(np.median(env_arr[q3:]))
            atk_start_lvl = low + 0.1 * (high - low)
            atk_end_lvl = low + 0.9 * (high - low)
            rel_start_lvl = high - 0.1 * (high - tail)
            rel_end_lvl = high - 0.9 * (high - tail)
            rise = env_arr[q1 - n // 10 : q3]
            fall = env_arr[q3 - n // 10 :]

            def _crossing(x: np.ndarray, level: float, direction: str = "up"):
                idxs = np.nonzero(x >= level)[0] if direction == "up" else np.nonzero(x <= level)[0]
                return int(idxs[0]) if idxs.size else None

            atk_start_rel = _crossing(rise, atk_start_lvl, "up")
            atk_end_rel = _crossing(rise, atk_end_lvl, "up")
            rel_start_rel = _crossing(fall, rel_start_lvl, "down")
            rel_end_rel = _crossing(fall, rel_end_lvl, "down")

            markers = {}
            if atk_start_rel is not None:
                markers["atk_start"] = (q1 - n // 10 + atk_start_rel) / fs
            if atk_end_rel is not None:
                markers["atk_end"] = (q1 - n // 10 + atk_end_rel) / fs
            if rel_start_rel is not None:
                markers["rel_start"] = (q3 - n // 10 + rel_start_rel) / fs
            if rel_end_rel is not None:
                markers["rel_end"] = (q3 - n // 10 + rel_end_rel) / fs
            return markers

        markers = _find_markers(env)
        for key, xt in markers.items():
            ax_env.axvline(xt, color="red", linestyle="--", alpha=0.6)
            ax_env.text(xt, max(env) * 0.8, key, rotation=90, va="bottom", ha="center", fontsize=8)

        ax_env.set_xlabel("Time (s)")
        ax_env.set_ylabel("RMS")
        ax_env.set_title(
            f"Attack {metrics.get('attack_ms', float('nan')):.1f} ms | Release {metrics.get('release_ms', float('nan')):.1f} ms"
        )
        ax_env.legend()

        def _export_csv():
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
            if not path:
                return
            import csv

            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time_s", "envelope"])
                for tt, vv in zip(env_t, env):
                    w.writerow([float(tt), float(vv)])
            self._log(f"Đã xuất CSV: {path}")

        ttk.Button(controls, text="Export CSV", command=_export_csv).pack(side="right", padx=6, pady=4)

        fig.tight_layout()
        canvas.draw()


plot_windows.PlotWindowManager = _PlotWindowManager

# ------------------------------ threading -----------------------------------
import threading as _threading_mod


def _run_in_thread(target: Callable, stop_event: _threading_mod.Event, name: Optional[str] = None) -> _threading_mod.Thread:
    thread = _threading_mod.Thread(target=target, name=name or target.__name__)
    thread.daemon = True
    stop_event.clear()
    thread.start()
    return thread


threading.run_in_thread = _run_in_thread

