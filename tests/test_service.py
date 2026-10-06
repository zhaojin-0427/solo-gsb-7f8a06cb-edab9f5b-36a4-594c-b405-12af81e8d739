"""端到端测试：使用临时 SQLite 文件与 FastAPI TestClient。

覆盖：
* 正弦波 rFFT 与 NumPy 参考实现一致
* 乱序 / 重叠 / 重复提交的幂等与拒绝语义
* 不同分块方式产生完全相同的频谱结果
* 配置临界切换：跨界窗口按旧配置、版本归属、乐观版本冲突
* 缺口不补零、封流越界拒绝、缺口与未满窗口列表
* 持久游标分页
* 并发提交结果确定
* 崩溃恢复（任务停在 running，重启续算，不重不漏）
"""
from __future__ import annotations

import os
import tempfile
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("SPECTRUM_DB", db_path)
    # 直接导入前固定环境变量
    import importlib
    from app import db as db_mod
    importlib.reload(db_mod)
    from app import main as main_mod
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        c.db_path = db_path
        c.main_mod = main_mod
        yield c


def _create(client, sr=1000.0, L=8, step=4):
    r = client.post("/streams", json={
        "sample_rate": sr, "window_len": L, "step": step})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _all_windows(client, sid, limit=1000):
    out, cursor = [], None
    while True:
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        r = client.get(f"/streams/{sid}/windows", params=params)
        assert r.status_code == 200
        body = r.json()
        out.extend(body["items"])
        if not body["has_more"]:
            break
        cursor = body["next_cursor"]
    return out


def _coeff_matrix(windows):
    order = sorted(windows, key=lambda w: w["start_sample"])
    return np.array([
        np.array(w["coeffs"]["real"]) + 1j * np.array(w["coeffs"]["imag"])
        for w in order
    ]), order


# ---------------------------------------------------------------------------
# 1. 正弦波：与 NumPy 参考实现一致
# ---------------------------------------------------------------------------

def test_sine_wave_matches_reference(client):
    sr, L, step = 256.0, 16, 8
    sid = _create(client, sr=sr, L=L, step=step)
    t = np.arange(40) / sr
    x = np.sin(2 * np.pi * 16.0 * t)  # 16 Hz，16 点恰为整数周期
    r = client.post(f"/streams/{sid}/blocks", json={
        "start_sample": 0, "values": x.tolist()})
    assert r.status_code == 200
    windows = _all_windows(client, sid)
    starts = [w["start_sample"] for w in windows]
    assert starts == list(range(0, 40 - L + 1, step))
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / L)
    for w in windows:
        s = w["start_sample"]
        ref = np.fft.rfft(x[s:s + L] * hann)
        got = (np.array(w["coeffs"]["real"])
               + 1j * np.array(w["coeffs"]["imag"]))
        assert np.allclose(ref, got, atol=1e-12)
        assert w["revision"] == 1
        assert w["end_sample"] == s + L
        assert len(w["frequencies"]) == L // 2 + 1
    # 16 Hz 处能量集中（bin 1）
    c0 = np.array(windows[0]["coeffs"]["real"]) + 1j * np.array(
        windows[0]["coeffs"]["imag"])
    mag = np.abs(c0)
    assert mag[1] == max(mag)


# ---------------------------------------------------------------------------
# 2. 乱序 / 重叠 / 重复：不同分块与顺序结果相同
# ---------------------------------------------------------------------------

def test_out_of_order_and_overlap_determinism(client):
    sr, L, step = 100.0, 8, 4
    rng = np.random.default_rng(42)
    x = rng.standard_normal(64)

    def run_feed(plan):
        sid = _create(client, sr=sr, L=L, step=step)
        for s, e in plan:
            r = client.post(f"/streams/{sid}/blocks", json={
                "start_sample": int(s), "values": x[s:e].tolist()})
            assert r.status_code == 200, (s, e, r.text)
        return sid

    # 顺序一：乱序的不相交小块
    p1 = [(32, 48), (0, 16), (48, 64), (16, 32)]
    # 顺序二：大块一次给
    p2 = [(0, 64)]
    # 顺序三：重叠、内容一致，顺序打乱
    p3 = [(40, 64), (0, 24), (16, 48)]

    mats = []
    for p in (p1, p2, p3):
        sid = run_feed(p)
        info = client.get(f"/streams/{sid}").json()
        assert info["prefix_end"] == 64
        mats.append(_coeff_matrix(_all_windows(client, sid))[0])

    assert np.array_equal(mats[0], mats[1])
    assert np.array_equal(mats[1], mats[2])


def test_duplicate_returns_original_result(client):
    sid = _create(client, L=8, step=4)
    x = np.linspace(0, 1, 16)
    r1 = client.post(f"/streams/{sid}/blocks",
                     json={"start_sample": 0, "values": x.tolist()})
    first = r1.json()
    r2 = client.post(f"/streams/{sid}/blocks",
                     json={"start_sample": 0, "values": x.tolist()})
    second = r2.json()
    assert second["duplicate"] is True
    assert second["block_id"] == first["block_id"]
    assert second["accepted_at"] == first["accepted_at"]
    info = client.get(f"/streams/{sid}").json()
    assert info["blocks_received"] == 1  # 账本只记一次


def test_conflicting_overlap_rejects_whole_block(client):
    sid = _create(client, L=8, step=4)
    r = client.post(f"/streams/{sid}/blocks",
                    json={"start_sample": 0, "values": [1.0] * 16})
    assert r.status_code == 200
    # 重叠区 [8,16) 的样本不同 -> 整块拒绝
    r = client.post(f"/streams/{sid}/blocks",
                    json={"start_sample": 8,
                          "values": [2.0] * 8 + [2.0] * 8})
    assert r.status_code == 409
    info = client.get(f"/streams/{sid}").json()
    assert info["blocks_received"] == 1
    # 重叠区一致、仅向外延伸的块正常接受（账本多一条，前缀推进）
    r = client.post(f"/streams/{sid}/blocks",
                    json={"start_sample": 8,
                          "values": [1.0] * 8 + [3.0] * 8})
    assert r.status_code == 200
    assert r.json()["prefix_end"] == 24
    # 完全一致的重叠范围 -> 视为重复
    r = client.post(f"/streams/{sid}/blocks",
                    json={"start_sample": 0, "values": [1.0] * 16})
    assert r.status_code == 200 and r.json()["duplicate"]


# ---------------------------------------------------------------------------
# 3. 缺口不补零，只推进连续前缀
# ---------------------------------------------------------------------------

def test_gap_does_not_auto_zero_fill(client):
    sid = _create(client, L=4, step=2)
    client.post(f"/streams/{sid}/blocks",
                json={"start_sample": 0, "values": [1.0, 1.0]})
    client.post(f"/streams/{sid}/blocks",
                json={"start_sample": 4, "values": [1.0, 1.0, 1.0, 1.0]})
    info = client.get(f"/streams/{sid}").json()
    assert info["prefix_end"] == 2  # 缺口 [2,4)，前缀停在 2
    windows = _all_windows(client, sid)
    assert windows == []  # 窗长 4，尚无完整窗口，绝不把缺口补零
    # 补齐缺口（内容由后续块给定）
    r = client.post(f"/streams/{sid}/blocks",
                    json={"start_sample": 2, "values": [0.0, 0.0]})
    assert r.json()["prefix_end"] == 8
    windows = _all_windows(client, sid)
    assert [w["start_sample"] for w in windows] == [0, 2, 4]


# ---------------------------------------------------------------------------
# 4. 配置临界切换
# ---------------------------------------------------------------------------

def test_config_critical_switch(client):
    # 旧：L=8 step=4；新版 L=16 step=8 生效样本 S=16
    sid = _create(client, sr=100.0, L=8, step=4)
    cfg = client.post(f"/streams/{sid}/configs", json={
        "expected_revision": 1, "window_len": 16, "step": 8,
        "effective_s": 16})
    assert cfg.status_code == 200
    assert cfg.json()["revision"] == 2

    # 版本冲突
    r = client.post(f"/streams/{sid}/configs", json={
        "expected_revision": 1, "window_len": 4, "step": 2,
        "effective_s": 32})
    assert r.status_code == 409

    # S 必须严格递增
    r = client.post(f"/streams/{sid}/configs", json={
        "expected_revision": 2, "window_len": 4, "step": 2,
        "effective_s": 16})
    assert r.status_code == 409

    x = np.arange(32, dtype=float)
    r = client.post(f"/streams/{sid}/blocks",
                    json={"start_sample": 0, "values": x.tolist()})
    assert r.status_code == 200
    windows = _all_windows(client, sid)
    specs = [(w["revision"], w["start_sample"], w["end_sample"])
             for w in windows]
    # 旧版起点 0,4,8,12（12+8=20 跨界但仍归旧版，按旧配置完成）
    # 新版起点 16,24（16+16=32, 24+16=40>32 不算）
    assert specs == [
        (1, 0, 8), (1, 4, 12), (1, 8, 16), (1, 12, 20),
        (2, 16, 32),
    ]
    # 窗口起点唯一，没有同窗两套配置
    starts = [s[1] for s in specs]
    assert len(starts) == len(set(starts))

    # 数值核对：跨界窗 12..20 必须是旧 L=8 窗
    hann8 = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(8) / 8)
    w12 = next(w for w in windows if w["start_sample"] == 12)
    got = np.array(w12["coeffs"]["real"]) + 1j * np.array(
        w12["coeffs"]["imag"])
    assert np.allclose(got, np.fft.rfft(x[12:20] * hann8))
    w16 = next(w for w in windows if w["start_sample"] == 16)
    assert w16["end_sample"] == 32 and len(w16["coeffs"]["real"]) == 9
    hann16 = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(16) / 16)
    got16 = np.array(w16["coeffs"]["real"]) + 1j * np.array(
        w16["coeffs"]["imag"])
    assert np.allclose(got16, np.fft.rfft(x[16:32] * hann16))


def test_config_effective_s_before_prefix_rejected(client):
    sid = _create(client, L=4, step=2)
    client.post(f"/streams/{sid}/blocks",
                json={"start_sample": 0, "values": [0.0] * 16})
    # prefix_end=16，S=8 早于前缀终点 -> 拒绝
    r = client.post(f"/streams/{sid}/configs", json={
        "expected_revision": 1, "window_len": 8, "step": 4,
        "effective_s": 8})
    assert r.status_code == 409
    # S=16 恰好不早于终点 -> 接受
    r = client.post(f"/streams/{sid}/configs", json={
        "expected_revision": 1, "window_len": 8, "step": 4,
        "effective_s": 16})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# 5. 封流：越界拒绝、缺口与未满窗口
# ---------------------------------------------------------------------------

def test_seal_rejects_when_data_exceeds_n(client):
    sid = _create(client, L=4, step=2)
    client.post(f"/streams/{sid}/blocks",
                json={"start_sample": 0, "values": [0.0] * 20})
    r = client.post(f"/streams/{sid}/seal", json={"total_samples": 16})
    assert r.status_code == 409
    info = client.get(f"/streams/{sid}").json()
    assert info["sealed"] is False


def test_seal_with_gap_lists_gaps_and_unfull_windows(client):
    sid = _create(client, L=8, step=4)
    # [0,10) 与 [14,24)，缺口 [10,14)
    client.post(f"/streams/{sid}/blocks",
                json={"start_sample": 0, "values": [1.0] * 10})
    client.post(f"/streams/{sid}/blocks",
                json={"start_sample": 14, "values": [1.0] * 10})
    r = client.post(f"/streams/{sid}/seal", json={"total_samples": 24})
    assert r.status_code == 200
    body = r.json()
    assert body["gaps"] == [{"start_sample": 10, "end_sample": 14}]
    assert body["prefix_end"] == 10
    # 已完成：只有窗 0（窗 4 需要样本到 12，但 [10,14) 有缺口不补零）
    assert [w["start_sample"] for w in body["completed_windows"]] == [0]
    # 未满窗口：起点在 [0,24) 且数据不足完成 -> 4(缺口卡住),8,12,16,20(尾窗越 N)
    assert [w["start_sample"] for w in body["unfull_windows"]] == [
        4, 8, 12, 16, 20]
    # 封流后拒绝新数据
    r = client.post(f"/streams/{sid}/blocks",
                    json={"start_sample": 10, "values": [1.0] * 4})
    assert r.status_code == 409
    r = client.post(f"/streams/{sid}/seal", json={"total_samples": 24})
    assert r.status_code == 409


def test_seal_clean_stream_completes_all_windows(client):
    sid = _create(client, L=8, step=4)
    x = np.arange(20, dtype=float)
    client.post(f"/streams/{sid}/blocks",
                json={"start_sample": 0, "values": x.tolist()})
    r = client.post(f"/streams/{sid}/seal", json={"total_samples": 20})
    body = r.json()
    assert body["gaps"] == []
    # 窗 16 起点在 [0,20) 内但需要样本到 24，永远未满
    assert [w["start_sample"] for w in body["unfull_windows"]] == [16]
    assert [w["start_sample"] for w in body["completed_windows"]] == \
        [0, 4, 8, 12]


# ---------------------------------------------------------------------------
# 6. 持久游标分页
# ---------------------------------------------------------------------------

def test_cursor_pagination(client):
    sid = _create(client, L=4, step=2)
    client.post(f"/streams/{sid}/blocks",
                json={"start_sample": 0, "values": [0.0] * 16})
    seen, cursor = [], None
    for _ in range(20):
        params = {"limit": 3}
        if cursor:
            params["cursor"] = cursor
        body = client.get(f"/streams/{sid}/windows", params=params).json()
        seen.extend(w["start_sample"] for w in body["items"])
        if not body["has_more"]:
            break
        cursor = body["next_cursor"]
    assert seen == [0, 2, 4, 6, 8, 10, 12]


# ---------------------------------------------------------------------------
# 7. 并发提交结果确定
# ---------------------------------------------------------------------------

def test_concurrent_feeds_deterministic(client):
    rng = np.random.default_rng(7)
    x = rng.standard_normal(80)
    sid = _create(client, L=8, step=4)

    def feed(ranges):
        for s, e in ranges:
            r = client.post(f"/streams/{sid}/blocks",
                            json={"start_sample": int(s),
                                  "values": x[s:e].tolist()})
            assert r.status_code in (200, 409)
    ranges = [(0, 20), (16, 40), (32, 56), (48, 72), (64, 80)]
    threads = [
        threading.Thread(target=feed, args=(list(ranges),)),
        threading.Thread(target=feed, args=(list(reversed(ranges)),)),
        threading.Thread(target=feed, args=(list(ranges),)),  # 重复
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    info = client.get(f"/streams/{sid}").json()
    assert info["prefix_end"] == 80
    mat, order = _coeff_matrix(_all_windows(client, sid))
    assert [w["start_sample"] for w in order] == list(range(0, 73, 4))
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(8) / 8)
    for i, w in enumerate(order):
        s = w["start_sample"]
        ref = np.fft.rfft(x[s:s + 8] * hann)
        assert np.allclose(mat[i], ref)


# ---------------------------------------------------------------------------
# 8. 崩溃恢复：任务停在 running，重启后续算，结果不重不漏
# ---------------------------------------------------------------------------

def test_crash_recovery(client, tmp_path, monkeypatch):
    sid = _create(client, L=8, step=4)
    x = np.arange(40, dtype=float)
    client.post(f"/streams/{sid}/blocks",
                json={"start_sample": 0, "values": x[:16].tolist()})
    before = len(_all_windows(client, sid))
    assert before == 3  # 0,4,8

    # 模拟崩溃：再喂数据后，手工把部分任务标记为 running（进程死在计算/提交中）
    from app.db import connect
    db_path = client.db_path
    client.post(f"/streams/{sid}/blocks",
                json={"start_sample": 16, "values": x[16:40].tolist()})
    conn = connect(db_path)
    conn.execute(
        "UPDATE window_tasks SET status='running', claim_token='dead'"
        " WHERE stream_id=? AND start_sample>=12", (sid,))
    conn.commit()
    conn.close()

    # 重启：全新 TestClient（等价于新进程，同一数据库文件）
    import importlib
    from app import main as main_mod
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c2:
        # 启动恢复应把 running 退回 pending 并续算
        windows = _all_windows(c2, sid)
        starts = [w["start_sample"] for w in windows]
        assert starts == list(range(0, 33, 4))
        info = c2.get(f"/streams/{sid}").json()
        assert info["windows_pending"] == 0
        assert info["prefix_end"] == 40
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(8) / 8)
        for w in windows:
            s = w["start_sample"]
            ref = np.fft.rfft(x[s:s + 8] * hann)
            got = np.array(w["coeffs"]["real"]) + 1j * np.array(
                w["coeffs"]["imag"])
            assert np.allclose(ref, got)
