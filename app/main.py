"""FastAPI 入口：连续采样频谱 HTTP 服务。"""
from __future__ import annotations

import math
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import service
from .db import connect, init_db, recover_on_startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    app.state.db_path = os.environ.get("SPECTRUM_DB", "/data/spectrum.db")
    conn = connect(app.state.db_path)
    init_db(conn)
    recover_on_startup(conn)
    service.resume_all_pending(conn)
    conn.close()
    yield


app = FastAPI(
    title="连续采样频谱服务",
    version="1.0.0",
    description="按流提交有限实数采样块，服务输出加周期 Hann 窗的 rFFT。",
    lifespan=lifespan,
)


def get_conn(request: Request):
    # 每请求一个 SQLite 连接（SQLite 连接不跨线程共享）。
    conn = connect(request.app.state.db_path)
    try:
        yield conn
    finally:
        conn.close()


@app.exception_handler(service.ServiceError)
async def service_error_handler(request: Request, exc: service.ServiceError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

class CreateStreamIn(BaseModel):
    sample_rate: float = Field(..., gt=0, description="固定采样率 Hz")
    window_len: int = Field(..., ge=2, description="初始窗长（样本数，>=2）")
    step: int = Field(..., ge=1, description="初始正整数步长")


class BlockIn(BaseModel):
    start_sample: int = Field(..., ge=0, description="块起始样本序号")
    values: list[float] = Field(..., description="有限实数样本")


class ConfigIn(BaseModel):
    expected_revision: int = Field(..., description="预期当前版本，乐观并发控制")
    window_len: int = Field(..., ge=2)
    step: int = Field(..., ge=1)
    effective_s: int = Field(..., ge=0, description="生效样本 S")


class SealIn(BaseModel):
    total_samples: int = Field(..., ge=0, description="总样本数 N")


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/streams", status_code=201)
def create_stream(body: CreateStreamIn, conn=Depends(get_conn)):
    return service.create_stream(
        conn, body.sample_rate, body.window_len, body.step
    )


@app.get("/streams/{sid}")
def get_stream(sid: str, conn=Depends(get_conn)):
    return service.get_stream(conn, sid)


@app.get("/streams/{sid}/configs")
def get_configs(sid: str, conn=Depends(get_conn)):
    return {"stream_id": sid, "configs": service.list_configs(conn, sid)}


@app.post("/streams/{sid}/blocks")
def post_block(sid: str, body: BlockIn, conn=Depends(get_conn)):
    if not body.values:
        return JSONResponse(status_code=400,
                            content={"detail": "values must be non-empty"})
    if any((isinstance(v, bool) or not math.isfinite(v)) for v in body.values):
        return JSONResponse(status_code=400,
                            content={"detail": "values must be finite"})
    return service.submit_block(conn, sid, body.start_sample, body.values)


@app.post("/streams/{sid}/configs")
def post_config(sid: str, body: ConfigIn, conn=Depends(get_conn)):
    return service.submit_config(
        conn, sid, body.expected_revision, body.window_len, body.step,
        body.effective_s,
    )


@app.get("/streams/{sid}/windows")
def get_windows(
    sid: str,
    conn=Depends(get_conn),
    cursor: str | None = Query(None, description="上次响应的 next_cursor"),
    limit: int = Query(100, ge=1, le=1000),
):
    after = service.decode_cursor(cursor)
    return service.list_windows(conn, sid, after, limit)


@app.post("/streams/{sid}/seal")
def post_seal(sid: str, body: SealIn, conn=Depends(get_conn)):
    return service.seal_stream(conn, sid, body.total_samples)
