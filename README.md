# Severe Convection Multi-Agent Decision System

> 强对流预警多智能体系统 — 可信辅助决策原型

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个面向气象预报员的**多智能体辅助决策系统**，用于强对流天气（短时强降水、大风、冰雹、龙卷风）的风险研判与决策支持。系统不是自动预警发布系统，而是帮助预报员快速完成多源产品研判、形成建议与审计闭环的工具。

## 核心特性

- **9 个专业智能体**协同推理：动力、环境、雷达、临近预报、客观指导、模式共识、数据质量、交叉一致性、LLM 态势
- **证据融合层**：支持代理源降权、相关性惩罚、per-agent 贡献分解
- **策略引擎**：issue/clear 双阈值、人工复核闸门、降级模式
- **LLM 集成**：支持 OpenAI (gpt-4o-mini) / DeepSeek，三种模式：off / shadow / assist
- **真值标签治理**：gold/silver/proxy 分层，proxy 不进入主评估与训练
- **评估治理**：时间切分 train/calibration/test，业务指标 + bootstrap CI
- **空间评估**：行政区级 + 网格级命中/偏差
- **审计回放**：run_id、字段级 provenance、决策链、回放与对比
- **自动窗口扫描**：筛选可评估的历史时段
- **权重自进化**：基于 Brier score 优化智能体权重 + 概率校准

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                     │
│          (Leaflet Map / Console / Ops Dashboard)         │
├─────────────────────────────────────────────────────────┤
│                    Orchestrator Layer                     │
│    (Agent Dispatch → Evidence Fusion → Policy Gate)      │
├─────────────────────────────────────────────────────────┤
│                      Agent Layer                         │
│  Circulation | Environment | Radar | Nowcast | Guidance  │
│  ModelConsistency | DataQuality | CrossSource | LLM      │
├─────────────────────────────────────────────────────────┤
│                     Domain Layer                         │
│  Observation | EvidenceCard | FusionResult | HazardObj   │
│  DecisionPacket | AuditRecord | PolicySnapshot           │
├─────────────────────────────────────────────────────────┤
│                  Data Ingestion Layer                     │
│  Open-Meteo | RainViewer | NOAA nowCOAST | NOAA ISD     │
│  Objective Guidance | Hail Reports | Area Resolver       │
├─────────────────────────────────────────────────────────┤
│                   Governance Layer                        │
│  Truth Labels | Evaluator | Evolver | Memory | Registry  │
│  Replay | Policy Engine | Migrations                     │
└─────────────────────────────────────────────────────────┘
```

## 快速启动

```powershell
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
$env:AGENT_PROFILE='dev'
$env:AGENT_API_TOKEN='your-token-here'

# 启动服务
python -m uvicorn weather_agent.api:app --host 127.0.0.1 --port 8000
```

浏览器访问：`http://127.0.0.1:8000/`

## 一键脚本

```powershell
./scripts/init_demo_data.ps1          # 初始化演示数据
./scripts/run_demo_infer.ps1          # 运行推理演示
./scripts/run_demo_eval.ps1           # 运行评估演示
./scripts/run_demo_evolve.ps1         # 运行权重进化演示
./scripts/generate_reports.ps1        # 生成报告
./scripts/migrate_storage.ps1         # 存储迁移
./scripts/run_nightly_regression.ps1  # 夜间回归测试
./scripts/find_positive_windows.ps1   # 自动筛选可评估历史窗口
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/forecast/live` | POST | 实时数据获取 + 推理 |
| `/api/v1/forecast/manual` | POST | 手动观测输入 |
| `/api/v1/evaluate/recent` | POST | 历史回放评估 |
| `/api/v1/evolve/weights` | POST | 权重优化 + 校准器训练 |
| `/api/v1/truth/build` | POST | 构建真值标签 |
| `/api/v1/windows/scan` | POST | 自动窗口扫描 |
| `/api/v1/audit/{run_id}` | GET | 审计记录查询 |
| `/api/v1/replay/case` | POST | 回放验证 |
| `/api/v1/runtime/llm` | GET/POST | LLM 运行时配置 |

完整 API 文档：启动服务后访问 `http://127.0.0.1:8000/docs`

## 技术栈

- **后端**：Python 3.10+ / FastAPI / Uvicorn / Pydantic v2
- **前端**：原生 HTML/CSS/JS + Leaflet 地图
- **数据源**：Open-Meteo (ECMWF/GFS/ICON) / NOAA nowCOAST / NOAA ISD / RainViewer
- **LLM**：OpenAI gpt-4o-mini / DeepSeek (可配置)
- **存储**：SQLite (registry) + JSON (runs/audit/memory)
- **测试**：pytest + GitHub Actions 夜间回归

## 重要声明

- 系统输出是**建议**，不是官方发布
- 当出现降级/冲突/数据过期时，系统会进入 `manual_review` 状态
- 代理源（如雷达瓦片代理、特征工程概率）会被显式标记并降权

## 文档

- [系统架构](docs/architecture.md)
- [评估治理](docs/evaluation_governance.md)
- [审计与回放](docs/audit_and_replay.md)
- [运维指南](docs/operator_guide.md)
- [风险声明](docs/risk_disclaimer.md)
- [技术路线说明](docs/系统技术路线全说明.md)

## License

MIT
