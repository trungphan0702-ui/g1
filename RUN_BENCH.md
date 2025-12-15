# Offline DSP Bench Harness

Use this harness to run the offline DSP benchmarks (THD/THD+N, compressor curve, attack/release, compare metrics) without the GUI or any audio hardware.

## 1. Chuẩn bị môi trường

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Chạy benchmark mẫu

```bash
python tools/run_offline_bench.py \
  --config tools/bench_config.json \
  --out out/bench_results.json \
  --csv out/bench_results.csv
```

Tùy chọn:
- Thêm `--verbose` hoặc đặt `DSP_DEBUG=1` để in log chi tiết.
- Chỉnh file `tools/bench_config.json` để thay đổi tham số, thêm đường dẫn WAV.

## 3. Output & cách gửi lại

Sau khi chạy, kết quả được ghi tại:
- `out/bench_results.json` – JSON tổng hợp (bao gồm thông tin môi trường).
- `out/bench_results.csv` – mỗi test case một dòng, mở được bằng Excel.
- Console sẽ in tóm tắt từng case (THD/THD+N, attack/release, compressor, compare).

Khi gửi lại cho tôi, vui lòng đính kèm:
1. File `bench_results.json`
2. File `bench_results.csv`
3. Console log (copy/paste) của lệnh trên.

## 4. Tùy chỉnh config nhanh

- **THD**: `thd_cases` cho phép chọn `type: synthetic` hoặc `type: file` (`input_wav`). Có thể thêm harmonic bằng `harmonic.order`, `harmonic.level_db`. Các tham số FFT: `thd_window`, `thd_nfft`, `thd_fund_band_bins`, `thd_max_h`.
- **Attack/Release**: set `freq`, `amp`, `duration`, `rms_win_ms` trong `attack_release_cases`.
- **Compressor**: `compressor_cases` hỗ trợ `apply_compressor: true/false` và các tham số threshold/ratio/makeup/knee/attack/release.
- **Compare**: đặt `input_wav` và `output_wav` cho từng case trong `compare_cases`.

## 5. Ví dụ output rút gọn

```
Wrote results to out/bench_results.json and out/bench_results.csv
- thd_clean_sine [thd] | THD -120.00 dB, THD+N -90.00 dB
- thd_distorted_sine [thd] | THD -20.12 dB, THD+N -19.50 dB
- attack_release_default [attack_release] | Attack 50.0 ms / Release 150.0 ms
- compressor_curve_default [compressor] | Thr nan dB, Ratio 1.0
- compressor_curve_with_model [compressor] | Thr -12.0 dB, Ratio 4.0
- compare_example [compare] | Skipped: missing input/output wav.
```

> Các con số trên chỉ là ví dụ minh hoạ, không phải kết quả thật.
