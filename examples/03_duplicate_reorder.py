"""示例 3：重复、乱序与内容一致的重叠。

* 乱序提交不相交块，服务只推进连续前缀；
* 内容一致的重叠块被接受（账本多一条，样本不重复计算）；
* 同范围完全相同的块作为重复返回原接收结果（block_id / accepted_at 不变）；
* 重叠样本只要有一个不一致，整块拒绝。

运行：
    SPECTRUM_URL=http://localhost:8000 python examples/03_duplicate_reorder.py
"""
from __future__ import annotations

import numpy as np
import requests

from common import BASE_URL


def post_block(sid, start, values):
    r = requests.post(f"{BASE_URL}/streams/{sid}/blocks", json={
        "start_sample": start, "values": list(values)}, timeout=10)
    return r


def main() -> None:
    r = requests.post(f"{BASE_URL}/streams", json={
        "sample_rate": 100.0, "window_len": 8, "step": 4}, timeout=10)
    sid = r.json()["id"]
    print(f"created stream {sid}")

    rng = np.random.default_rng(0)
    x = rng.standard_normal(48)

    # 乱序：后半段先来
    r = post_block(sid, 32, x[32:48]); r.raise_for_status()
    print(f"out-of-order [32,48): prefix_end={r.json()['prefix_end']} (gap)")

    # 内容一致的重叠：[0,20) 与 [16,40)
    r = post_block(sid, 0, x[0:20]); r.raise_for_status()
    print(f"[0,20): prefix_end={r.json()['prefix_end']}")
    r = post_block(sid, 16, x[16:40]); r.raise_for_status()
    print(f"consistent overlap [16,40): prefix_end={r.json()['prefix_end']}")
    r = post_block(sid, 24, x[24:48]); r.raise_for_status()
    print(f"consistent overlap [24,48): prefix_end={r.json()['prefix_end']}")

    # 重复提交同范围同内容 -> 返回原始接收结果
    r1 = post_block(sid, 0, x[0:20])
    r2 = post_block(sid, 0, x[0:20])
    j1, j2 = r1.json(), r2.json()
    assert j2["duplicate"] is True
    assert j2["block_id"] == j1["block_id"]
    assert j2["accepted_at"] == j1["accepted_at"]
    print(f"duplicate: block_id={j2['block_id']} accepted_at unchanged")

    # 部分重叠但重叠样本不一致 -> 整块 409
    bad = x[4:24].copy()
    bad[0] += 1.0  # 样本 4 与既有块 [0,20) 不一致
    r = post_block(sid, 4, bad)
    assert r.status_code == 409
    print(f"conflicting partial overlap rejected: {r.json()['detail']}")

    info = requests.get(f"{BASE_URL}/streams/{sid}").json()
    print(f"ledger: blocks={info['blocks_received']} "
          f"prefix_end={info['prefix_end']} windows={info['windows_done']}")

    # 乱序 + 重叠产生的频谱与一次整块上传一致
    sid2 = requests.post(f"{BASE_URL}/streams", json={
        "sample_rate": 100.0, "window_len": 8, "step": 4}).json()["id"]
    post_block(sid2, 0, x).raise_for_status()

    def coeffs(s):
        items, cursor = [], None
        while True:
            params = {"limit": 100}
            if cursor:
                params["cursor"] = cursor
            b = requests.get(f"{BASE_URL}/streams/{s}/windows",
                             params=params).json()
            items.extend(b["items"])
            if not b["has_more"]:
                break
            cursor = b["next_cursor"]
        return items

    a, b = coeffs(sid), coeffs(sid2)
    assert len(a) == len(b)
    for wa, wb in zip(a, b):
        assert wa["start_sample"] == wb["start_sample"]
        assert np.allclose(wa["coeffs"]["real"], wb["coeffs"]["real"])
        assert np.allclose(wa["coeffs"]["imag"], wb["coeffs"]["imag"])
    print(f"spectra identical across chunkings/order ({len(a)} windows)")


if __name__ == "__main__":
    main()
