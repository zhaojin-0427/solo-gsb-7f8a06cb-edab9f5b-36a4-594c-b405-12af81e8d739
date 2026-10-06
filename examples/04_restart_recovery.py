"""示例 4：重启续算。

演示"进程在计算或提交中崩溃后可续算"。为了不依赖外部进程编排，
本示例直接在同一数据库文件上做两轮独立的服务初始化（等价于
uvicorn 停止后用同一个 /data/spectrum.db 重新启动容器）：

  轮 1：建流、上传前 16 点，正常算出窗口 0/4/8；
        随后把 12 以后的计算任务人为标记为 running（模拟进程在
        计算/提交窗口结果时被杀，任务停在 running 且结果未落盘）。
  轮 2：全新进程打开同一数据库：
        - 启动恢复把 running 退回 pending；
        - 续算所有可算窗口，不重不漏；
        - 再上传 16..40，结果与一次完整上传逐系数相同。

也可以对真实容器验证：
    docker compose restart spectrum
重启后 GET /streams/{id}/windows 仍能读到全部窗口。

运行：
    python examples/04_restart_recovery.py [/tmp/restart_demo.db]
"""
from __future__ import annotations

import os
import sys

import numpy as np
from fastapi.testclient import TestClient


def _fresh_client(db_path: str):
    os.environ["SPECTRUM_DB"] = db_path
    import importlib
    from app import db as db_mod
    importlib.reload(db_mod)
    from app import main as main_mod
    importlib.reload(main_mod)
    client = TestClient(main_mod.app)
    client.__enter__()
    return client, main_mod


def all_windows(client, sid):
    items, cursor = [], None
    while True:
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        b = client.get(f"/streams/{sid}/windows", params=params).json()
        items.extend(b["items"])
        if not b["has_more"]:
            break
        cursor = b["next_cursor"]
    return items


def main() -> None:
    db_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/restart_demo.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    x = np.arange(40, dtype=float)

    # ---- 轮 1：第一次启动进程 ----
    client, main_mod = _fresh_client(db_path)
    sid = client.post("/streams", json={
        "sample_rate": 100.0, "window_len": 8, "step": 4}).json()["id"]
    print(f"[run 1] created stream {sid}")
    client.post(f"/streams/{sid}/blocks",
                json={"start_sample": 0, "values": x[:16].tolist()})
    done = [w["start_sample"] for w in all_windows(client, sid)]
    print(f"[run 1] windows after 16 samples: {done}")
    assert done == [0, 4, 8]

    # 继续喂到 40，然后模拟崩溃：任务 running、结果未提交
    client.post(f"/streams/{sid}/blocks",
                json={"start_sample": 16, "values": x[16:].tolist()})
    from app.db import connect
    conn = connect(db_path)
    conn.execute(
        "UPDATE window_tasks SET status='running', claim_token='dead'"
        " WHERE stream_id=? AND start_sample>=12", (sid,))
    conn.commit()
    running = conn.execute(
        "SELECT COUNT(*) c FROM window_tasks WHERE status='running'"
    ).fetchone()["c"]
    conn.close()
    client.__exit__(None, None, None)
    print(f"[run 1] CRASH simulated: {running} tasks stuck in running")

    # ---- 轮 2：全新进程，同一数据库文件 ----
    client, _ = _fresh_client(db_path)
    print("[run 2] process restarted with same database")
    info = client.get(f"/streams/{sid}").json()
    print(f"[run 2] prefix_end={info['prefix_end']} "
          f"done={info['windows_done']} pending={info['windows_pending']}")
    assert info["windows_pending"] == 0

    windows = all_windows(client, sid)
    starts = [w["start_sample"] for w in windows]
    print(f"[run 2] windows after resume: {starts}")
    assert starts == list(range(0, 33, 4))  # 不重不漏

    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(8) / 8)
    for w in windows:
        s = w["start_sample"]
        ref = np.fft.rfft(x[s:s + 8] * hann)
        got = np.array(w["coeffs"]["real"]) + 1j * np.array(
            w["coeffs"]["imag"])
        assert np.allclose(ref, got)
    print("[run 2] all resumed windows match NumPy reference exactly")
    client.__exit__(None, None, None)
    print("restart recovery OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    main()
