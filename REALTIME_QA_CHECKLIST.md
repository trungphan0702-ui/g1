# Realtime Audio QA Checklist (Soundcard Loopback)

This checklist guides a live soundcard loopback session (Windows-friendly) using the existing GUI or any sounddevice-compatible script. Follow the steps in order; after each stage, capture the requested artifacts so results can be reviewed and tuned.

## 0) Collect environment info (tell me before running tests)
- Audio backend/library: `sounddevice` / ASIO / WASAPI / other? Include version if known.
- Device list output and selected IDs/names for **Input** and **Output**.
- Current settings: samplerate (`fs`), channels, buffer/blocksize or latency settings (if set by the driver/app).

## 1) Smoke test (prove I/O path works)
1. Enumerate devices. With sounddevice: `python -m sounddevice` or `python - <<'PY'\nimport sounddevice as sd; print(sd.query_devices())\nPY`.
2. Select **one** input and **one** output device ID. Confirm they share a common `default_samplerate` (e.g., 48000 Hz).
3. Playback-only sanity: generate a -18 dBFS 1 kHz sine for 2 s; ensure you can hear it without distortion.
4. Record-only sanity: record silence for 2 s; verify the file is not clipped and the noise floor is reasonable (< -70 dBFS typical for consumer cards).
5. Loopback sanity: play -18 dBFS 1 kHz for 2 s and record; verify the recorded file shows the tone at roughly the same level and no heavy clicks. Save as `smoke_loop.wav`.

Send back: device list, chosen input/output, smoke test console logs, and `smoke_loop.wav` path.

## 2) Test A – Compressor LIVE (Stepped sweep)
- **Stimulus**: 1 kHz sine, 48 kHz fs, mono, 0.25 s per step, 18 steps from -45 dBFS to -3 dBFS (linear steps of 2.5 dB). Apply 10 ms fade-in/out per step.
- **Playback level**: start at -45 dBFS; ensure output chain does not clip. Increase only if noise floor is too high.
- **Record**: mono, 48 kHz, blocksize 256–512 if adjustable, latency/buffer default. Save TX as `comp_tx.wav` (optional) and RX as `comp_rx.wav`.
- **Analysis**: segment RX by steps; compute RMS per step; fit compressor curve using `compressor.compression_curve`. Extract: threshold (dBFS), ratio, gain_offset_db, `no_compression`, and per-step points.
- **Expected ranges**: if no compressor, ratio ≈1 and `no_compression=True`; with gentle compression, threshold somewhere between -24 and -12 dBFS, ratio 2–4:1; makeup gain shows as positive gain offset.
- **Common issues**: clipping near -3 dBFS, noise gates causing truncated low-level steps, AGC causing slow drift.

Send back: console log, fitted params, step-by-step RMS table/plot (IO curve), and `comp_rx.wav` path (plus `comp_tx.wav` if saved).

## 3) Test B – THD LIVE
- **Stimulus**: pure 1 kHz sine, 48 kHz fs, mono, 2.0 s, level -6 dBFS (start at -12 dBFS if unsure). Apply 10 ms fade-in/out.
- **Record**: mono, 48 kHz. Save TX as `thd_tx.wav` (optional) and RX as `thd_rx.wav`.
- **Analysis**: use `thd.compute_thd` with `hmax=5`, Blackman-Harris or Hann window, nfft >= 32768. Report: thd_db, thdn_db, thd_percent, harmonics_dbc (H2–H5), fund_freq, fs, nfft, window, fund_band_bins.
- **Expected behavior**: clean interface should show THD well below -80 dB (≈0.01%); harmonics typically H2/H3 dominate. If leakage appears, confirm windowing and exact bin centering (fundamental at ~1000 Hz).

Send back: console log, THD metrics table, spectrum plot (mark H1–H5), and `thd_rx.wav` path (plus `thd_tx.wav` if saved).

## 4) Test C – Attack/Release LIVE
- **Stimulus**: level step tone from -30 dBFS (0.8 s) → -6 dBFS (0.8 s) → -30 dBFS (0.8 s); 1 kHz, 48 kHz fs, mono. Fade edges 10 ms.
- **Record**: mono, 48 kHz. Save TX as `ar_tx.wav` (optional) and RX as `ar_rx.wav`.
- **Analysis**: compute RMS envelope (e.g., 5 ms window). Measure attack time 10%→90% on rising edge, release time 90%→10% on falling edge using `attack_release.attack_release_times` or `compare_attack_release`. Report attack_ms, release_ms and window size used.
- **Expected**: clean path shows very fast (<10 ms) envelope transitions. With compressor/limiter, attack may be 1–50 ms; release can range 50–500 ms depending on settings. Look for overshoot or hold periods indicating lookahead/limiting.

Send back: console log, measured attack/release times, RMS envelope plot with markers, and `ar_rx.wav` path (plus `ar_tx.wav` if saved).

## 5) Test D – Loopback WAV (single file)
- Choose one WAV input file (`loop_in.wav`). Play it out through the device chain and record RX as `loop_rx.wav`.
- Align TX/RX: use `compare.align_signals` to estimate latency; trim RX accordingly; apply `compare.gain_match` before analysis.
- Run analyses on RX:
  - Compressor analysis (if the file is a stepped sweep)
  - THD analysis (if tone-based)
  - Attack/Release analysis (if step-based)
- Report latency_ms, gain_error_db, and relevant metrics. Note any pre-roll/post-roll silence you trimmed.

Send back: console log, measured latency/gain error, analysis results, and both file paths (`loop_in.wav`, `loop_rx.wav`).

## 6) Test E – Offline compare (no live loopback)
- Provide two WAVs (`ref.wav`, `test.wav`).
- Use `compare.align_signals`, `compare.gain_match`, and `compare.residual_metrics`.
- Report: latency_ms, gain_error_db, noise_floor_dbfs, residual_rms_dbfs, snr_db, thd_ref_db, thd_tgt_db, thd_delta_db, fr_dev_median_db, hum_peaks.

Send back: console log, metrics summary, and the two file paths.

## Notes and safety
- Start levels conservatively (-18 to -12 dBFS) and raise only if the noise floor is too high. Watch for clipping LEDs or flat-topped waveforms.
- If using `sounddevice`, some versions expect `device=(out_id, in_id)` while others use `output_device`/`input_device`. Match to your installed version.
- Keep sample rate consistent (48 kHz recommended). Use mono to simplify alignment unless your device requires stereo.
- If latency compensation exists in the driver, still report the measured latency from recorded files—it helps verify alignment.

## What to send me after each stage
- Console logs
- Measured values (threshold/ratio, THD numbers, attack/release times, residual metrics)
- Saved WAV paths (TX and RX if available)
- Screenshots of plots: spectrum with harmonic markers, IO curve for compressor, RMS envelope for attack/release

I will review each stage and provide tuned next steps based on your results.
