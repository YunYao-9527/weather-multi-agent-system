# 强对流多智能体联调版（前后端 + 真实数据 + 评估演进）

本工程已从 MVP 升级为“可联调演示版”，满足以下目标：

1. 前端联调版：可直接打开网页，具备智能体工程风格的控制台。
2. 后端联调规范：CORS、Bearer 鉴权、错误码、固定响应 schema。
3. 真实数据接入：模式/客观概率/雷达适配器已串联到 `adapters/`。
4. 月份与灾种阈值：按季节差异写入 Agent 规则（冰雹/雷暴大风/短时强降水）。
5. 可评估可优化：支持 `baseline vs enhanced`、近期回放评估、自主演进权重更新。
6. 可信度增强：新增数据质量建模、跨源一致性Agent、概率校准训练与在线校准。

## 一、快速启动

安装依赖：

```powershell
pip install -r requirements.txt
```

启动服务：

```powershell
python -m uvicorn weather_agent.api:app --host 127.0.0.1 --port 8000
```

打开网页：

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

默认 Token：

- `agent-dev-token`

## 二、联调规范

### 1) 鉴权

- Header: `Authorization: Bearer <token>`
- 环境变量：`AGENT_API_TOKEN`（默认 `agent-dev-token`）
- 可通过 `AGENT_ENABLE_AUTH=0` 关闭（开发调试）

### 2) CORS

- 环境变量：`AGENT_CORS_ORIGINS`，默认 `*`

### 3) 固定响应 schema

成功：

```json
{
  "code": "OK",
  "message": "success",
  "request_id": "uuid",
  "data": {}
}
```

失败：

```json
{
  "code": "AUTH_001",
  "message": "Missing Bearer token",
  "request_id": "uuid",
  "detail": null
}
```

### 4) 错误码约定

- `AUTH_001`：缺少 Bearer Token
- `AUTH_002`：Token 非法
- `REQ_001`：请求参数校验失败
- `DATA_001`：外部数据拉取失败
- `SYS_001`：系统内部异常

## 三、接口列表

- `GET /api/v1/health`
- `GET /api/v1/geo/city?city=<name>`
- `POST /api/v1/forecast/manual`
- `POST /api/v1/forecast/live`
- `POST /api/v1/evaluate/recent`
- `POST /api/v1/evolve/weights`
- `POST /api/v1/truth/build`

说明：

- `geo/city`：按输入城市返回经纬度与时区，供前端地图联动。
- `forecast/live`：实时拉取外部数据并完成一次协同推理；当 `area` 为空且 `auto_area=true` 时，系统按“经纬度缓冲 + 行政区匹配”自动生成落区。
- `evaluate/recent`：近 N 天历史回放评估，优先使用真值标签管线（站点雨强/风/冰雹报告），并输出覆盖率与审计信息。
- `evolve/weights`：基于历史回放优化 agent 权重，并写入长期记忆。
- `truth/build`：按自定义时间窗构建真值标签产物，落盘并附带 SHA256 便于审计复现。

## 四、真实数据来源（已接入）

### 1) 模式与历史回放（Open-Meteo）

文件：

- `weather_agent/adapters/open_meteo.py`
- `weather_agent/adapters/live_snapshot.py`
- `weather_agent/evaluator.py`
- `weather_agent/evolver.py`

用途：

- 多模式（`ecmwf_ifs025`、`gfs_seamless`、`icon_seamless`）实时特征
- 历史归档（archive）用于回放评估与权重优化

### 2) 雷达（RainViewer）

文件：

- `weather_agent/adapters/rainviewer.py`

用途：

- 拉取最新雷达图层并采样目标点，生成雷达强度代理特征

### 3) 客观概率（业务代理实现）

文件：

- `weather_agent/adapters/objective_guidance.py`

用途：

- 基于模式特征生成短时强降水/大风/冰雹/龙卷概率，作为客观 guidance

## 五、月份与灾种阈值规则（已落地）

文件：

- `weather_agent/rules.py`
- `weather_agent/agents/environment.py`
- `weather_agent/agents/radar.py`

规则概念：

1. 4-5 月（春季切变型）：较低 CAPE + 较强切变的冰雹/大风风险更敏感。
2. 7-8 月（夏季高能型）：高 CAPE 与高湿背景，短时强降水权重更高。
3. 其他月份：使用折中阈值。

灾种维度：

- 冰雹：`WBZ`、雷达阈值、CAPE、切变
- 大风：`DCAPE`、`T850-500`、切变、低层湿度
- 短时强降水：低层湿度、CAPE、低层辐合

## 六、显式计划 + 长期记忆 + 自主演进

### 1) 显式计划（Plan）

文件：

- `weather_agent/planner.py`

作用：

- 每次推理返回固定步骤计划，便于解释与审计。

### 2) 长期记忆（Memory）

文件：

- `weather_agent/memory.py`

作用：

- 按 `city:month` 存储和读取权重，形成“按月经验沉淀”。

### 3) 自主演进（Evolve）

文件：

- `weather_agent/evolver.py`

作用：

- 基于近期回放误差反推各 Agent 质量，自动更新融合权重。
- 同时训练灾种概率校准器（histogram calibrator），降低概率偏差。

### 4) 可信度增强（Confidence Upgrade）

新增能力：

1. 数据质量评分（`data_quality_score`）：模型覆盖、雷达时效、模式离散度综合。
2. 跨源一致性评分：环境场、客观概率、雷达信号三方一致性。
3. 置信度重构：由“仅看分歧”升级为“证据强度 + 一致性 + 数据质量”综合。
4. 在线概率校准：演进后加载校准器，对融合概率做后验校正。

新增 Agent：

- `data_quality_guard`
- `cross_source_consistency`

## 七、评估与目标对比

评估接口输出：

- `baseline` / `enhanced`
- 指标：`precision`、`recall`、`f1`、`brier`
- 改进摘要：`improvements.hazards.*`
- 是否达成“>=10%”目标：`overall_target_10pct_met`
- 演进接口输出包含 `calibrators`，并持久化到 `memory/prob_calibrators.json`
- 真值标签信息：`truth_labels.coverage_ratio/artifact_path/sha256/station_info`

当前版本评估结论通常会出现：

1. 风险稀有事件导致 F1 在短窗期可能为 0。
2. Brier 改善先出现于部分灾种（如风），再逐步扩展到其他灾种。
3. 若未达到 10%，说明还需补充更强监督标签与更高质量雷达/实况特征。

## 八、网页控制台功能

网页入口：

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

支持：

1. 实时推理（`forecast/live`）
2. 近 3 天评估（`evaluate/recent`）
3. 自主演进优化（`evolve/weights`）
4. 输入城市地图实时联动（`geo/city`）
5. 请求处理中转圈反馈与按钮禁用，避免重复触发
6. 评估与演进自动生成文字总结解释
7. 一键构建指定时间窗真值标签（`truth/build`）
8. 查看显式计划、灾种概率条、证据链、评估结果、演进权重、原始响应

说明：

- 由于 NOAA 站点数据存在发布滞后，若某一时段站点数据尚未落库，系统会在评估中自动回退代理标签并在 `truth_labels` 字段中标注覆盖率与回退说明。

## 九、目录速览

- `weather_agent/api.py`：API、CORS、鉴权、错误码、schema
- `weather_agent/adapters/`：外部数据适配层
- `weather_agent/agents/`：多 Agent 业务规则
- `weather_agent/rules.py`：按月份和灾种阈值
- `weather_agent/planner.py`：显式计划
- `weather_agent/memory.py`：长期记忆
- `weather_agent/evaluator.py`：回放评估
- `weather_agent/evolver.py`：权重自演进
- `web/`：前端联调页面
### 4) 真值标签管线（可审计）

文件：

- `weather_agent/adapters/noaa_isd.py`（NOAA ISD 站点小时数据）
- `weather_agent/adapters/hail_reports.py`（本地冰雹报告 CSV）
- `weather_agent/truth_labels.py`（标签构建与审计产物）

说明：

1. 雨强标签：站点小时雨强阈值（默认 20 mm/h）。
2. 大风标签：站点小时极大风阈值（默认 17.2 m/s）。
3. 冰雹标签：站点现象码 + 本地冰雹报告文件融合。
4. 每次构建生成 `runs/truth_labels/*.json`，包含标签覆盖率、站点列表、SHA256。
