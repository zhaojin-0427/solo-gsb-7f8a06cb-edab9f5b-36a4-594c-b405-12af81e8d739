"""示例 1：正弦波。

建流（固定采样率 + 初始窗长/步长），上传一段 50 Hz 正弦波，
按持久游标读出 rFFT 频谱，并找出谱峰。

运行：
    SPECTRUM_URL=http://localhost:8000 python examples/01_sine_wave.py
"""
from __future__ import annotations

import numpy as np
import requests

from common import BASE_URL


def main() -> None:
    sr = 1000.0          # 固定采样率
    L, step = 64, 32     # 初始窗长与步长（从样本 0 生效）
    n = 256

    r = requests.post(f"{BASE_URL}/streams", json={
        "sample_rate": sr, "window_len": L, "step": step}, timeout=10)
    r.raise_for_status()
    sid = r.json()["id"]
    print(f"created stream {sid}: fs={sr} Hz, L={L}, step={step}")

    t = np.arange(n) / sr
    # 62.5 Hz 在 L=64、fs=1000 时恰好落在第 4 个 bin（1000/64*4）
    x = np.sin(2 * np.pi * 62.5 * t) + 0.3 * np.sin(2 * np.pi * 200.0 * t)
    r = requests.post(f"{BASE_URL}/streams/{sid}/blocks", json={
        "start_sample": 0, "values": x.tolist()}, timeout=30)
    r.raise_for_status()
    print(f"block accepted, prefix_end={r.json()['prefix_end']}")

    # 用持久游标逐页读取
    cursor, count = None, 0
    peaks = []
    while True:
        params = {"limit": 10}
        if cursor:
            params["cursor"] = cursor
        body = requests.get(
            f"{BASE_URL}/streams/{sid}/windows", params=params, timeout=10
        ).json()
        for w in body["items"]:
            count += 1
            c = np.array(w["coeffs"]["real"]) + 1j * np.array(
                w["coeffs"]["imag"])
            k = int(np.argmax(np.abs(c)))
            peaks.append(w["frequencies"][k])
            print(
                f"window rev={w['revision']} "
                f"samples=[{w['start_sample']},{w['end_sample']}) "
                f"peak={w['frequencies'][k]:.2f} Hz"
            )
        if not body["has_more"]:
            break
        cursor = body["next_cursor"]
    print(f"read {count} windows; expected peak 62.5 Hz, observed "
          f"{np.median(peaks):.1f} Hz")

    r = requests.post(f"{BASE_URL}/streams/{sid}/seal",
                      json={"total_samples": n}, timeout=10)
    r.raise_for_status()
    print(f"sealed: gaps={r.json()['gaps']}, "
          f"unfull={[w['start_sample'] for w in r.json()['unfull_windows']]}")


if __name__ == "__main__":
    main()
