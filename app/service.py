"""领域服务：接收账本、连续前缀推进、配置版本、窗口规划与 rFFT 计算。

设计要点
========
* 所有状态变更都在 ``BEGIN IMMEDIATE`` 事务中完成，SQLite 保证写事务
  全局唯一先后 —— 配置提交与样本接收无论怎么并发，都有确定的序列化顺序。
* blocks 是"接收账本"：原样保存每个被接受块的 float64 字节，
  重叠内容逐字节比对；segments 是去重后物化的连续前缀分片，
  二者保证不同分块/上传顺序得到相同字节序列。
* window_tasks 采用 claim 模式（pending→running→done），
  结果与 done 状态同一事务提交，崩溃后 running 回退 pending 重新计算，
  因此不漏窗、不重窗、计算可重试且结果确定。
"""
from __future__ import annotations

import hashlib
import math
import uuid
from datetime import datetime, timezone

import numpy as np

from .db import immediate


class ServiceError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 流
# ---------------------------------------------------------------------------

def create_stream(conn, sample_rate: float, window_len: int, step: int) -> dict:
    sid = uuid.uuid4().hex
    ts = _now()
    with immediate(conn):
        conn.execute(
            "INSERT INTO streams(id, sample_rate, created_at) VALUES (?,?,?)",
            (sid, float(sample_rate), ts),
        )
        # 初始配置：流内 revision=1，生效样本 S=0
        conn.execute(
            "INSERT INTO configs(stream_id, revision, window_len, step,"
            " effective_s, created_at) VALUES (?,?,?,?,?,?)",
            (sid, 1, int(window_len), int(step), 0, ts),
        )
    return get_stream(conn, sid)


def get_stream(conn, sid: str) -> dict:
    row = conn.execute("SELECT * FROM streams WHERE id=?", (sid,)).fetchone()
    if row is None:
        raise ServiceError(404, "stream not found")
    cfg = conn.execute(
        "SELECT revision, window_len, step, effective_s FROM configs"
        " WHERE stream_id=? ORDER BY revision DESC LIMIT 1",
        (sid,),
    ).fetchone()
    n_blocks = conn.execute(
        "SELECT COUNT(*) c FROM blocks WHERE stream_id=?", (sid,)
    ).fetchone()["c"]
    n_windows = conn.execute(
        "SELECT COUNT(*) c FROM window_results WHERE stream_id=?", (sid,)
    ).fetchone()["c"]
    n_pending = conn.execute(
        "SELECT COUNT(*) c FROM window_tasks WHERE stream_id=? AND status!='done'",
        (sid,),
    ).fetchone()["c"]
    return {
        "id": row["id"],
        "sample_rate": row["sample_rate"],
        "prefix_end": row["prefix_end"],
        "sealed": bool(row["sealed"]),
        "total_samples": row["total_samples"],
        "current_revision": cfg["revision"],
        "window_len": cfg["window_len"],
        "step": cfg["step"],
        "effective_s": cfg["effective_s"],
        "blocks_received": n_blocks,
        "windows_done": n_windows,
        "windows_pending": n_pending,
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# 配置版本
# ---------------------------------------------------------------------------

def list_configs(conn, sid: str) -> list[dict]:
    _require_stream(conn, sid)
    rows = conn.execute(
        "SELECT revision, window_len, step, effective_s, created_at"
        " FROM configs WHERE stream_id=? ORDER BY revision",
        (sid,),
    ).fetchall()
    return [dict(r) for r in rows]


def submit_config(
    conn, sid: str, expected_revision: int, window_len: int, step: int,
    effective_s: int,
) -> dict:
    with immediate(conn):
        st = _locked_stream(conn, sid)
        if st["sealed"]:
            raise ServiceError(409, "stream is sealed")
        cur = conn.execute(
            "SELECT * FROM configs WHERE stream_id=? ORDER BY revision DESC LIMIT 1",
            (sid,),
        ).fetchone()
        if expected_revision != cur["revision"]:
            raise ServiceError(
                409,
                f"revision conflict: expected {cur['revision']}, "
                f"got {expected_revision}",
            )
        if effective_s <= cur["effective_s"]:
            raise ServiceError(
                409,
                f"effective_s must be strictly greater than previous "
                f"{cur['effective_s']}",
            )
        if effective_s < st["prefix_end"]:
            raise ServiceError(
                409,
                f"effective_s {effective_s} is earlier than current "
                f"prefix_end {st['prefix_end']}",
            )
        ts = _now()
        cur_rev = cur["revision"] + 1
        conn.execute(
            "INSERT INTO configs(stream_id, revision, window_len, step,"
            " effective_s, created_at) VALUES (?,?,?,?,?,?)",
            (sid, cur_rev, int(window_len), int(step), int(effective_s), ts),
        )
        # 新版生效后，旧版窗口区间被钉死，立即规划当前数据已能支撑的部分。
        _plan_tasks_locked(conn, sid, st["prefix_end"])

    run_pending(conn, sid)
    return {
        "revision": cur_rev,
        "window_len": int(window_len),
        "step": int(step),
        "effective_s": int(effective_s),
        "created_at": ts,
    }


# ---------------------------------------------------------------------------
# 样本块提交
# ---------------------------------------------------------------------------

def submit_block(conn, sid: str, start_sample: int, values: list[float]) -> dict:
    start_sample = int(start_sample)
    if start_sample < 0:
        raise ServiceError(400, "start_sample must be >= 0")
    if not values:
        raise ServiceError(400, "values must be non-empty")
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ServiceError(400, "values must be finite real numbers")

    arr = np.asarray(values, dtype=np.float64)
    blob = arr.tobytes()
    digest = hashlib.sha256(blob).hexdigest()
    end_sample = start_sample + len(values)
    ts = _now()

    with immediate(conn):
        st = _locked_stream(conn, sid)
        if st["sealed"]:
            raise ServiceError(409, "stream is sealed")

        overlaps = conn.execute(
            "SELECT * FROM blocks WHERE stream_id=? "
            "AND start_sample < ? AND end_sample > ?"
            " ORDER BY start_sample",
            (sid, end_sample, start_sample),
        ).fetchall()

        duplicate_row = None
        for row in overlaps:
            if row["start_sample"] == start_sample and row["end_sample"] == end_sample:
                if row["content_hash"] == digest:
                    duplicate_row = row
                    break
                raise ServiceError(
                    409,
                    f"block at {start_sample} same range but different content; "
                    f"whole block rejected",
                )
            # 重叠但范围不同：逐样本比对，任何不一致则整块拒绝。
            old = np.frombuffer(row["data"], dtype=np.float64)
            a = max(start_sample, row["start_sample"])
            b = min(end_sample, row["end_sample"])
            new_part = arr[a - start_sample:b - start_sample]
            old_part = old[a - row["start_sample"]:b - row["start_sample"]]
            if not np.array_equal(new_part, old_part, equal_nan=False):
                raise ServiceError(
                    409,
                    f"overlapping samples [{a},{b}) disagree with block "
                    f"{row['id']}; whole block rejected",
                )

        if duplicate_row is not None:
            # 重复提交：不写入任何新数据，原样返回首次接收结果。
            result = {
                "block_id": duplicate_row["id"],
                "duplicate": True,
                "accepted": True,
                "start_sample": duplicate_row["start_sample"],
                "end_sample": duplicate_row["end_sample"],
                "accepted_at": duplicate_row["accepted_at"],
            }
        else:
            bid = conn.execute(
                "INSERT INTO blocks(stream_id, start_sample, end_sample, n_values,"
                " content_hash, data, accepted_at) VALUES (?,?,?,?,?,?,?)",
                (sid, start_sample, end_sample, len(values), digest, blob, ts),
            ).lastrowid
            hwm = _advance_prefix_locked(conn, sid, st["prefix_end"])
            if hwm != st["prefix_end"]:
                conn.execute(
                    "UPDATE streams SET prefix_end=? WHERE id=?", (hwm, sid)
                )
            _plan_tasks_locked(conn, sid, hwm)
            result = {
                "block_id": bid,
                "duplicate": False,
                "accepted": True,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "accepted_at": ts,
            }

        hwm_now = conn.execute(
            "SELECT prefix_end FROM streams WHERE id=?", (sid,)
        ).fetchone()["prefix_end"]
        result["prefix_end"] = hwm_now

    # 事务外做数值计算，避免长事务；崩溃安全由 claim 账本保证。
    run_pending(conn, sid)
    return result


def _advance_prefix_locked(conn, sid: str, hwm: int) -> int:
    """只沿"已连续收到的前缀"推进；缺口存在时绝不补零、绝不跨越。

    每轮选一个起点贴住当前前缀且能扩展的块（选择规则确定：
    end 最大、start 最小、id 最小），把新增 [hwm, end) 物化为 segment。
    重叠块内容一致（提交时已校验），选谁产生的字节都相同。
    """
    while True:
        row = conn.execute(
            "SELECT * FROM blocks WHERE stream_id=? AND start_sample <= ? "
            "AND end_sample > ? ORDER BY end_sample DESC, start_sample ASC,"
            " id ASC LIMIT 1",
            (sid, hwm, hwm),
        ).fetchone()
        if row is None:
            return hwm
        off = (hwm - row["start_sample"]) * 8
        new_blob = row["data"][off:off + (row["end_sample"] - hwm) * 8]
        conn.execute(
            "INSERT INTO segments(stream_id, start_sample, end_sample, data)"
            " VALUES (?,?,?,?)",
            (sid, hwm, row["end_sample"], new_blob),
        )
        hwm = row["end_sample"]


# ---------------------------------------------------------------------------
# 窗口规划与计算
# ---------------------------------------------------------------------------

def _versions_locked(conn, sid: str) -> list:
    return conn.execute(
        "SELECT revision, window_len, step, effective_s FROM configs"
        " WHERE stream_id=? ORDER BY revision",
        (sid,),
    ).fetchall()


def _plan_tasks_locked(conn, sid: str, hwm: int) -> int:
    """按版本规划"数据已连续可算"的窗口。任务主键 (stream, start) 去重，
    重复规划被 INSERT OR IGNORE 吸收，因此任何调用顺序都不重不漏。

    版本 i 拥有起点满足 S_i <= start < S_{i+1} 的窗口；
    跨界窗口（start 在旧版、end 跨入新版）仍归旧版、按旧配置算。
    """
    versions = _versions_locked(conn, sid)
    inserted = 0
    for i, v in enumerate(versions):
        rev, length, step, s0 = v["revision"], v["window_len"], v["step"], v["effective_s"]
        kmax_data = (hwm - length - s0) // step  # 要求 S+k*step+L <= hwm
        if i + 1 < len(versions):
            s_next = versions[i + 1]["effective_s"]
            kmax_cfg = (s_next - 1 - s0) // step
            kmax = min(kmax_data, kmax_cfg)
        else:
            kmax = kmax_data
        if kmax < 0:
            continue
        rows = [
            (sid, s0 + k * step, s0 + k * step + length, rev)
            for k in range(kmax + 1)
        ]
        for j in range(0, len(rows), 4096):
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO window_tasks"
                "(stream_id, start_sample, end_sample, revision)"
                " VALUES (?,?,?,?)",
                rows[j:j + 4096],
            )
            inserted += max(0, conn.total_changes - before)
    return inserted


def resume_all_pending(conn) -> int:
    """启动恢复：对所有流续算崩溃前未完成的窗口。"""
    sids = [r["stream_id"] for r in conn.execute(
        "SELECT DISTINCT stream_id FROM window_tasks WHERE status!='done'"
    ).fetchall()]
    total = 0
    for sid in sids:
        total += run_pending(conn, sid)
    return total


def _read_window_samples(conn, sid: str, start: int, end: int) -> np.ndarray:
    """从去重后的连续前缀分片拼出 [start,end)。segments 是 [0,hwm) 的
    一个不相交划分，调用方保证 end <= prefix_end。"""
    parts: list[bytes] = []
    rows = conn.execute(
        "SELECT start_sample, end_sample, data FROM segments WHERE stream_id=?"
        " AND start_sample < ? AND end_sample > ? ORDER BY start_sample",
        (sid, end, start),
    ).fetchall()
    for r in rows:
        a = max(start, r["start_sample"])
        b = min(end, r["end_sample"])
        off = (a - r["start_sample"]) * 8
        parts.append(r["data"][off:off + (b - a) * 8])
    raw = b"".join(parts)
    if len(raw) != (end - start) * 8:
        # 理论上不可达：任务只在 end<=prefix_end 时被领取。
        raise ServiceError(500, "internal error: contiguous samples missing")
    return np.frombuffer(raw, dtype=np.float64)


def _hann_periodic(length: int) -> np.ndarray:
    # 周期 Hann（等价 scipy.signal.windows.hann(M, sym=False)）
    n = np.arange(length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)


def run_pending(conn, sid: str, batch: int = 512) -> int:
    """领取并计算所有当前可算窗口。可被任意线程/进程重复调用：
    claim 用条件 UPDATE 保证一个任务只被一个执行者领取。"""
    computed = 0
    while True:
        with immediate(conn):
            rows = conn.execute(
                "SELECT start_sample, end_sample, revision FROM window_tasks"
                " WHERE stream_id=? AND status='pending' AND end_sample <="
                " (SELECT prefix_end FROM streams WHERE id=?)"
                " ORDER BY start_sample LIMIT ?",
                (sid, sid, batch),
            ).fetchall()
            if not rows:
                break
            token = uuid.uuid4().hex
            claimed = []
            for r in rows:
                cur = conn.execute(
                    "UPDATE window_tasks SET status='running', claim_token=?"
                    " WHERE stream_id=? AND start_sample=? AND status='pending'",
                    (token, sid, r["start_sample"]),
                )
                if cur.rowcount == 1:
                    claimed.append(r)
            st = conn.execute(
                "SELECT sample_rate FROM streams WHERE id=?", (sid,)
            ).fetchone()
            sample_rate = st["sample_rate"]

        payload = []
        for r in claimed:
            samples = _read_window_samples(
                conn, sid, r["start_sample"], r["end_sample"]
            )
            length = r["end_sample"] - r["start_sample"]
            coeffs = np.fft.rfft(samples * _hann_periodic(length))
            payload.append(
                (
                    sid,
                    r["start_sample"],
                    r["end_sample"],
                    r["revision"],
                    np.ascontiguousarray(coeffs, dtype=np.complex128).tobytes(),
                )
            )

        with immediate(conn):
            for p in payload:
                conn.execute(
                    "INSERT OR IGNORE INTO window_results"
                    "(stream_id, start_sample, end_sample, revision, coeffs)"
                    " VALUES (?,?,?,?,?)",
                    p,
                )
                conn.execute(
                    "UPDATE window_tasks SET status='done', claim_token=NULL"
                    " WHERE stream_id=? AND start_sample=?",
                    (sid, p[1]),
                )
        computed += len(payload)
        if len(rows) < batch:
            break
    return computed


# ---------------------------------------------------------------------------
# 结果读取（持久游标）
# ---------------------------------------------------------------------------

def list_windows(conn, sid: str, after: int, limit: int) -> dict:
    st = _require_stream(conn, sid)
    rows = conn.execute(
        "SELECT start_sample, end_sample, revision, coeffs FROM window_results"
        " WHERE stream_id=? AND start_sample > ? ORDER BY start_sample LIMIT ?",
        (sid, after, limit + 1),
    ).fetchall()
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = []
    for r in rows:
        c = np.frombuffer(r["coeffs"], dtype=np.complex128)
        length = r["end_sample"] - r["start_sample"]
        freqs = (np.arange(len(c), dtype=np.float64)
                 * st["sample_rate"] / length).tolist()
        items.append({
            "revision": r["revision"],
            "start_sample": r["start_sample"],
            "end_sample": r["end_sample"],  # 半开区间
            "coeffs": {
                "real": c.real.astype(float).tolist(),
                "imag": c.imag.astype(float).tolist(),
            },
            "frequencies": freqs,
        })
    next_after = rows[-1]["start_sample"] if rows else after
    return {
        "stream_id": sid,
        "items": items,
        "next_cursor": _encode_cursor(next_after) if (has_more or items) else None,
        "has_more": has_more,
    }


def _encode_cursor(after: int) -> str:
    import base64
    return base64.urlsafe_b64encode(str(int(after)).encode()).decode()


def decode_cursor(cursor: str | None) -> int:
    if not cursor:
        return -1
    import base64
    try:
        return int(base64.urlsafe_b64decode(cursor.encode()).decode())
    except Exception:
        raise ServiceError(400, "invalid cursor")


# ---------------------------------------------------------------------------
# 封流
# ---------------------------------------------------------------------------

def seal_stream(conn, sid: str, total_samples: int) -> dict:
    total_samples = int(total_samples)
    if total_samples < 0:
        raise ServiceError(400, "total_samples must be >= 0")
    with immediate(conn):
        st = _locked_stream(conn, sid)
        if st["sealed"]:
            raise ServiceError(409, "stream already sealed")
        # 已有数据越界则拒绝封流。
        bad = conn.execute(
            "SELECT COUNT(*) c FROM blocks WHERE stream_id=? AND "
            "(start_sample >= ? OR end_sample > ?)",
            (sid, total_samples, total_samples),
        ).fetchone()["c"]
        if bad:
            raise ServiceError(
                409,
                f"received blocks exceed declared total_samples={total_samples}; "
                f"seal rejected",
            )
        conn.execute(
            "UPDATE streams SET sealed=1, total_samples=? WHERE id=?",
            (total_samples, sid),
        )
        hwm = st["prefix_end"]
        _plan_tasks_locked(conn, sid, hwm)

        # [0,N) 内缺口：块区间并集的补集。
        gaps = _gaps_locked(conn, sid, total_samples)
        partial = _partial_windows_locked(
            conn, _versions_locked(conn, sid), hwm, total_samples
        )
    run_pending(conn, sid)
    with immediate(conn):
        done = conn.execute(
            "SELECT start_sample, end_sample, revision FROM window_results"
            " WHERE stream_id=? AND start_sample < ? ORDER BY start_sample",
            (sid, total_samples),
        ).fetchall()
    return {
        "stream_id": sid,
        "total_samples": total_samples,
        "prefix_end": hwm,
        "sealed": True,
        "gaps": gaps,
        "unfull_windows": partial,
        "completed_windows": [
            {
                "revision": r["revision"],
                "start_sample": r["start_sample"],
                "end_sample": r["end_sample"],
            }
            for r in done
        ],
    }


def _gaps_locked(conn, sid: str, n: int) -> list[dict]:
    if n == 0:
        return []
    rows = conn.execute(
        "SELECT start_sample, end_sample FROM blocks WHERE stream_id=?"
        " AND start_sample < ? AND end_sample > 0 ORDER BY start_sample",
        (sid, n),
    ).fetchall()
    gaps = []
    cur = 0
    for r in rows:
        a, b = max(0, r["start_sample"]), min(n, r["end_sample"])
        if a > cur:
            gaps.append({"start_sample": cur, "end_sample": a})
        cur = max(cur, b)
    if cur < n:
        gaps.append({"start_sample": cur, "end_sample": n})
    return gaps


def _partial_windows_locked(conn, versions, hwm: int, n: int) -> list[dict]:
    """起点在 [0,N) 内、但数据未连续到足以完成的窗口。"""
    out = []
    for i, v in enumerate(versions):
        length, step, s0 = v["window_len"], v["step"], v["effective_s"]
        if i + 1 < len(versions):
            kmax_cfg = (versions[i + 1]["effective_s"] - 1 - s0) // step
        else:
            kmax_cfg = (n - 1 - s0) // step
        if kmax_cfg < 0:
            continue
        # 起点在 [0,N)；未满即 end > hwm。
        k_lo = max(0, (hwm - length - s0) // step + 1)
        for k in range(k_lo, kmax_cfg + 1):
            start = s0 + k * step
            if start >= n:
                break
            out.append({
                "revision": v["revision"],
                "start_sample": start,
                "end_sample": start + length,
            })
    return out


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _require_stream(conn, sid: str):
    row = conn.execute("SELECT * FROM streams WHERE id=?", (sid,)).fetchone()
    if row is None:
        raise ServiceError(404, "stream not found")
    return row


def _locked_stream(conn, sid: str):
    """写事务内取流行并加表级写锁（IMMEDIATE 已保证串行）。"""
    return _require_stream(conn, sid)
