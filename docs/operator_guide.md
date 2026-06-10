# Operator Guide

## 1. 启动
```powershell
pip install -r requirements.txt
$env:AGENT_PROFILE='dev'
$env:AGENT_API_TOKEN='agent-dev-token'
python -m uvicorn weather_agent.api:app --host 127.0.0.1 --port 8000
```

## 2. 基本流程
1. 先构建真值（历史窗口）
2. 运行评估（建议 `truth_policy=require`）
3. 运行演进（仅在覆盖率达标时）
4. 实时推理与人工复核
5. 保存 run_id 并做审计

## 3. 页面使用
- 状态带：正常/降级/需人工复核
- Hazard Probability：灾种概率
- Evidence Readiness：证据完备度与冲突
- 审计面板：决策链、贡献分解、provenance

## 4. 常见问题
- `TRUTH_001`：真值覆盖不足，需换历史窗口或降低最小覆盖阈值（谨慎）
- `AUTH_001/002`：Token 问题
- `DATA_001`：实时数据源异常，检查网络或切换到手动输入

## 5. 推荐运行策略
- 业务评估：`truth_policy=require`
- 研发探索：`truth_policy=prefer` + 明确标注 auxiliary
- 正式研判：出现 `manual_review` 必须人工复核后再作业务动作

## 6. 雷达接入优先级
- 默认优先级：`local_grid -> nowcoast -> rainviewer`
- 可通过环境变量覆盖：
  - `AGENT_RADAR_PROVIDER_PRIORITY=local_grid,nowcoast,rainviewer`
  - `AGENT_RADAR_GRID_FILE=data/radar_grids/latest.json`
  - `AGENT_RADAR_GRID_MAX_DISTANCE_KM=180`
- 当仅命中 RainViewer 时，系统会标记 `radar_proxy_source=1`，融合层自动降权。

本地格点文件示例（JSON）：
```json
{
  "source": "business_radar_grid",
  "generated_at": "2026-04-02T13:40:00+08:00",
  "cells": [
    {"lat": 39.10, "lon": 117.20, "dbz": 46.0, "echo_top_km": 9.2, "vil": 24.3}
  ]
}
```

## 7. Nightly 回归
- 手动执行：`./scripts/run_nightly_regression.ps1`
- 默认开启门禁（`--enforce-gate`），覆盖率/样本数/Brier 不达标将失败退出。
- 产物路径：`runs/nightly/nightly_*.json|md`

## 8. 自动筛选可评估窗口
- 一键筛选：`./scripts/find_positive_windows.ps1 -City Tianjin -SearchStart 2025-03-01 -SearchEnd 2025-10-31 -WindowDays 3`
- 产物：`runs/window_scan/window_scan_*.json|md`
- 用法：先筛窗，再把推荐窗口填给 `/api/v1/evaluate/recent` 与 `/api/v1/evolve/weights`
- 建议门槛：`min_total_positive_labels >= 1`（至少有正例），业务评估可提高到 `>= 5` 或更高
- 可直接走 API：`POST /api/v1/windows/scan`
