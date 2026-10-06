# 连续采样频谱 HTTP 服务

Python 3.11 / FastAPI / NumPy / SQLite 实现。按流（stream）持续接收按样本序号定位的
有限实数采样块，只沿**已连续收到的前缀**推进，对每个完整窗口施加
**周期 Hann 窗**后计算 **rFFT**，复数系数以版本号和半开样本范围标注并持久化，
通过持久游标分页读取。

## 快速开始（Docker Compose）

```bash
docker compose up --build
# 服务监听 http://localhost:8000 ，交互式文档 /docs
# SQLite 数据文件存放在命名卷 spectrum-data -> 容器内 /data/spectrum.db
```

容器重启（`docker compose restart` 或宿主机重启）后自动续算：崩溃时停在
`running` 的窗口任务在启动时退回 `pending` 并重新计算，已提交的结果不重算，
不漏窗、不重复出窗。

## 从建流到读取频谱的调用过程

### 1. 建流：固定采样率 + 从样本 0 生效的初始窗长与步长

```bash
curl -s -X POST localhost:8000/streams \
  -H 'Content-Type: application/json' \
  -d '{"sample_rate": 1000.0, "window_len": 64, "step": 32}'
```

```json
{
  "id": "6733ffac…", "sample_rate": 1000.0,
  "prefix_end": 0, "sealed": false, "total_samples": null,
  "current_revision": 1, "window_len": 64, "step": 32, "effective_s": 0,
  "blocks_received": 0, "windows_done": 0, "windows_pending": 0
}
```

采样率在建流后固定。校验：`window_len >= 2`、`step >= 1`。
初始配置为**修订 1，生效样本 S=0**。

### 2. 提交采样块（可乱序、可重叠）

```bash
SID=6733ffac…
curl -s -X POST localhost:8000/streams/$SID/blocks \
  -H 'Content-Type: application/json' \
  -d '{"start_sample": 0, "values": [0.1, 0.2, ...]}'
```

* 值必须是**有限实数**（NaN/Inf 拒绝，400），块非空，`start_sample >= 0`。
* 允许**乱序**：缺口前方的块先存账本，不影响前缀。
* 允许**内容一致的重叠**：逐样本比对，重叠区全部一致才接受。
* **重复**（同范围且内容一致）不重复记账，原样返回首次接收的
  `block_id` 与 `accepted_at`，响应带 `"duplicate": true`。
* 任何重叠样本不一致 → **整块拒绝（409）**，不写入任何数据。
* 响应中的 `prefix_end` 是当前已连续前缀的终点（开区间）。
  **缺口不会自动补零、不会被跨越**：例如只收到 `[0,2)` 与 `[4,8)`，
  `prefix_end` 停在 2，窗长 4 时一个窗口都不会产生。

### 3. 按持久游标读取频谱

```bash
curl -s "localhost:8000/streams/$SID/windows?limit=100"
# 下一页带上响应里的 next_cursor
curl -s "localhost:8000/streams/$SID/windows?limit=100&cursor=OA"
```

每个窗口：

```json
{
  "revision": 1,
  "start_sample": 32,
  "end_sample": 96,
  "coeffs": {"real": [ … ], "imag": [ … ]},
  "frequencies": [0.0, 15.625, …]
}
```

* `[start_sample, end_sample)` 为**半开样本范围**，长度即该窗修订的窗长。
* `coeffs` 为 `np.fft.rfft` 的 `complex128` 系数，拆成实部/虚部数组，
  长度 `window_len/2 + 1`；`frequencies[k] = k * sample_rate / window_len`。
* 窗口按起点排序；游标是不透明字符串，基于"已返回最后起点"，
  重启后仍然有效。新算好的窗口在后续翻页中出现，永不重复。

### 4. 提交配置修订（可选，可多次）

```bash
curl -s -X POST localhost:8000/streams/$SID/configs \
  -H 'Content-Type: application/json' \
  -d '{"expected_revision": 1, "window_len": 128,
       "step": 64, "effective_s": 1024}'
```

* `window_len >= 2`、`step >= 1`。
* `expected_revision` 必须等于当前最新修订，否则 **409 冲突**（乐观并发控制）。
* `effective_s`（S）必须**严格大于**上一版的 S；且**不早于当前连续前缀终点**
  （`S >= prefix_end`），否则 409。
* **窗口归属**：修订 i 只拥有起点满足 `S_i <= start < S_{i+1}` 的窗口，
  起点为 `S_i + k*step`。**跨界窗口**（起点在旧版、末端跨入新版区间，
  例如旧 L=8 的窗 `[12,20)` 在 S=16 处跨界）仍归旧版、按旧配置完整计算。
* 配置提交与样本接收在 `BEGIN IMMEDIATE` 写事务下全局串行，有唯一先后：
  无论谁先提交，窗口都不会漏、不会重复、不会被算成两套配置。

### 5. 封流

```bash
curl -s -X POST localhost:8000/streams/$SID/seal \
  -H 'Content-Type: application/json' -d '{"total_samples": 256}'
```

* 已有任何块越过 `[0,N)` → **拒绝封流（409）**。
* 成功后流被冻结：新块、新配置、重复封流一律 409。
* 响应列出：
  * `gaps`：`[0,N)` 内相对已接收块并集的缺口（半开区间列表）；
  * `unfull_windows`：**起点在 `[0,N)` 内**但数据不足以完成的窗口
    （含因缺口卡住的窗，以及末端越过 N 永远无法完成的窗），带修订号；
  * `completed_windows`：已完成窗口的修订与范围。

## 运行示例

```bash
pip install requests        # 仅示例脚本需要
SPECTRUM_URL=http://localhost:8000 python examples/01_sine_wave.py        # 正弦波 + 谱峰
SPECTRUM_URL=http://localhost:8000 python examples/02_config_switch.py    # 配置临界切换/冲突
SPECTRUM_URL=http://localhost:8000 python examples/03_duplicate_reorder.py# 乱序/重叠/重复
python examples/04_restart_recovery.py                                   # 崩溃后续算（进程内）
```

## 测试

```bash
pip install -r requirements.txt pytest httpx
PYTHONPATH=. pytest -q
```

覆盖：正弦波对 NumPy 参考实现、乱序/重叠/重复幂等、不同分块结果逐系数一致、
缺口不补零、配置临界切换与冲突、封流越界与缺口/未满窗口、游标分页、
并发喂数确定性、任务停在 `running` 后的重启续算。

## 设计与持久化边界

| 组件 | 说明 |
| --- | --- |
| `blocks` | **接收账本**：每个被接受的块原样存 float64 小端字节与 sha256；重叠比对以此为准。重复不新增记录。 |
| `streams.prefix_end` | 已连续前缀终点（开区间）。提交块后在同一写事务内沿账本确定性推进（选择规则：end 最大、start 最小、id 最小），把新增部分物化进 `segments`。 |
| `segments` | `[0, prefix_end)` 的不相交分片，窗口计算的唯一数据源；任何分块/顺序产生相同字节。 |
| `configs` | 流内修订号（从 1 开始），含窗长、步长、生效样本 S。 |
| `window_tasks` | 计算账本：`pending → running(running 带 claim_token) → done`。规划用 `INSERT OR IGNORE` 按 `(stream, start)` 去重，任意调用顺序不重不漏。 |
| `window_results` | complex128 交错字节，与 `done` 在**同一事务**提交；崩溃在提交前则重启重算，提交后永不重算。 |

崩溃恢复（进程在计算或提交结果时被杀）：

1. 启动时 `recover_on_startup` 将所有 `running` 任务无条件退回 `pending`；
2. `resume_all_pending` 对每条流领取并续算；claim 用条件 UPDATE 保证
   多执行者不会重复领取；结果行用 `INSERT OR IGNORE` 保证幂等。

## API 摘要

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/streams` | 建流（采样率、初始窗长/步长） |
| GET | `/streams/{id}` | 状态（前缀终点、封流、版本、计数） |
| POST | `/streams/{id}/blocks` | 提交样本块 |
| GET | `/streams/{id}/configs` | 列全部配置修订 |
| POST | `/streams/{id}/configs` | 提交修订（expected_revision + S） |
| GET | `/streams/{id}/windows?cursor=&limit=` | 持久游标读取 rFFT 窗口 |
| POST | `/streams/{id}/seal` | 封流并声明 N，返回缺口与未满窗口 |
| GET | `/health` | 健康检查 |
