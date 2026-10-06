"""示例 2：配置临界切换。

初始配置 L=8 step=4；在样本 16 处切换为 L=16 step=8。
观察跨界窗口（起点 12，区间跨入新版生效点）仍按旧配置完成，
而起点 16 的窗口归新配置；同时演示版本冲突拒绝。

运行：
    SPECTRUM_URL=http://localhost:8000 python examples/02_config_switch.py
"""
from __future__ import annotations

import numpy as np
import requests

from common import BASE_URL


def main() -> None:
    r = requests.post(f"{BASE_URL}/streams", json={
        "sample_rate": 100.0, "window_len": 8, "step": 4}, timeout=10)
    sid = r.json()["id"]
    print(f"created stream {sid}: rev=1 L=8 step=4 S=0")

    # 先把数据送到 15：此时 S=16 不早于 prefix_end（S>=16），可以提交修订
    x = np.arange(15, dtype=float)
    requests.post(f"{BASE_URL}/streams/{sid}/blocks", json={
        "start_sample": 0, "values": x.tolist()}).raise_for_status()

    r = requests.post(f"{BASE_URL}/streams/{sid}/configs", json={
        "expected_revision": 1, "window_len": 16, "step": 8,
        "effective_s": 16}, timeout=10)
    r.raise_for_status()
    print(f"config updated -> rev={r.json()['revision']} S=16 L=16 step=8")

    # 再用过期版本号提交 -> 冲突
    r = requests.post(f"{BASE_URL}/streams/{sid}/configs", json={
        "expected_revision": 1, "window_len": 4, "step": 2,
        "effective_s": 32}, timeout=10)
    assert r.status_code == 409
    print(f"stale revision rejected: {r.json()['detail']}")

    # S 必须严格递增
    r = requests.post(f"{BASE_URL}/streams/{sid}/configs", json={
        "expected_revision": 2, "window_len": 4, "step": 2,
        "effective_s": 16}, timeout=10)
    assert r.status_code == 409
    print(f"non-increasing S rejected: {r.json()['detail']}")

    # 补齐到 32 个样本
    x = np.arange(32, dtype=float)
    requests.post(f"{BASE_URL}/streams/{sid}/blocks", json={
        "start_sample": 15, "values": x[15:].tolist()}).raise_for_status()

    body = requests.get(f"{BASE_URL}/streams/{sid}/windows",
                        params={"limit": 100}).json()
    for w in body["items"]:
        tag = "old" if w["revision"] == 1 else "new"
        print(f"  [{tag}] rev={w['revision']} "
              f"samples=[{w['start_sample']},{w['end_sample']})")
    specs = [(w["revision"], w["start_sample"], w["end_sample"])
             for w in body["items"]]
    assert specs == [
        (1, 0, 8), (1, 4, 12), (1, 8, 16), (1, 12, 20),   # 12 跨界仍归旧版
        (2, 16, 32),                                       # 新版第一窗
    ]
    print("assertions ok: cross-boundary window 12 used old L=8")


if __name__ == "__main__":
    main()
