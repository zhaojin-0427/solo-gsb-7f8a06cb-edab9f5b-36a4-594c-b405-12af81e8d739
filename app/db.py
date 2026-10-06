"""SQLite 持久层：连接、建表、低级事务辅助。

所有写操作都在 ``BEGIN IMMEDIATE`` 事务中串行执行，
这是本服务"配置提交与样本接收并发时有唯一先后"的根本保证。
"""
from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

# 允许用环境变量覆盖数据库路径，便于测试与容器挂载。
DB_PATH = os.environ.get("SPECTRUM_DB", "/data/spectrum.db")


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(
        path, timeout=30.0, isolation_level=None, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


@contextmanager
def immediate(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """开启写事务。SQLite 保证同一时刻只有一个 IMMEDIATE 事务。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


SCHEMA = """
CREATE TABLE IF NOT EXISTS streams (
    id            TEXT PRIMARY KEY,
    sample_rate   REAL NOT NULL,
    prefix_end    INTEGER NOT NULL DEFAULT 0,  -- 已连续收到的前缀终点（开区间）
    sealed        INTEGER NOT NULL DEFAULT 0,
    total_samples INTEGER,                    -- 封流后声明的 N
    created_at    TEXT NOT NULL
);

-- 接收账本：每一个被接受的块都留痕。
CREATE TABLE IF NOT EXISTS blocks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id    TEXT NOT NULL REFERENCES streams(id),
    start_sample INTEGER NOT NULL,
    end_sample   INTEGER NOT NULL,           -- 开区间
    n_values     INTEGER NOT NULL,
    content_hash TEXT NOT NULL,              -- float64 小端字节的 sha256
    data         BLOB NOT NULL,              -- 原始 float64 小端字节
    accepted_at  TEXT NOT NULL
    -- 不按内容哈希去重：不同范围但数值相同的块（如静音段）合法；
    -- 重复定义为"同范围且内容一致"，在提交逻辑中识别。
);

-- 去重后的样本载体：按已连续前缀物化的不相交段（blob 为 float64）。
CREATE TABLE IF NOT EXISTS segments (
    stream_id    TEXT NOT NULL REFERENCES streams(id),
    start_sample INTEGER NOT NULL,
    end_sample   INTEGER NOT NULL,
    data         BLOB NOT NULL,
    PRIMARY KEY (stream_id, start_sample)
);

-- 配置版本。revision=1 为建流时的初始配置（S=0），每流独立计数。
CREATE TABLE IF NOT EXISTS configs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id   TEXT NOT NULL REFERENCES streams(id),
    revision    INTEGER NOT NULL,         -- 流内版本号，从 1 开始
    window_len  INTEGER NOT NULL,
    step        INTEGER NOT NULL,
    effective_s INTEGER NOT NULL,         -- 生效样本 S
    created_at  TEXT NOT NULL,
    UNIQUE(stream_id, revision)
);

-- 计算任务账本（claim 模式），与结果表一一对应。
CREATE TABLE IF NOT EXISTS window_tasks (
    stream_id    TEXT NOT NULL,
    start_sample INTEGER NOT NULL,
    end_sample   INTEGER NOT NULL,
    revision     INTEGER NOT NULL,        -- 引用 configs.revision（流内版本号）
    status       TEXT NOT NULL DEFAULT 'pending', -- pending / running / done
    claim_token  TEXT,
    PRIMARY KEY (stream_id, start_sample),
    FOREIGN KEY (stream_id, revision) REFERENCES configs(stream_id, revision)
);

CREATE TABLE IF NOT EXISTS window_results (
    stream_id    TEXT NOT NULL REFERENCES streams(id),
    start_sample INTEGER NOT NULL,
    end_sample   INTEGER NOT NULL,
    revision     INTEGER NOT NULL,
    coeffs       BLOB NOT NULL,            -- 交错存储的 complex128 rFFT 系数
    PRIMARY KEY (stream_id, start_sample)
);

CREATE INDEX IF NOT EXISTS idx_tasks_status
    ON window_tasks(stream_id, status, start_sample);
CREATE INDEX IF NOT EXISTS idx_results_stream
    ON window_results(stream_id, start_sample);
CREATE INDEX IF NOT EXISTS idx_blocks_stream
    ON blocks(stream_id, start_sample, end_sample);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def recover_on_startup(conn: sqlite3.Connection) -> None:
    """进程在计算/提交中崩溃后：把 running 任务退回 pending。

    续算本身由 service.resume_all_pending() 完成（需要领域逻辑）。
    """
    conn.execute(
        "UPDATE window_tasks SET status='pending', claim_token=NULL "
        "WHERE status='running'"
    )
    conn.commit()
