# Architecture

## 1. 系统分层
1. 数据接入层：Open-Meteo（模式）、local radar grid/volume + NOAA nowCOAST（雷达实源）+ RainViewer（代理回退）、NOAA ISD + 本地冰雹报告（真值）
2. 领域层：Observation / EvidenceCard / FusionResult / WarningDecision / AuditRecord
3. Agent 层：circulation、environment、objective_guidance、model_consensus、radar、nowcast、data_quality_guard、cross_source_consistency
4. Orchestrator 层：统一调度、异常处理、融合、闸门决策、审计产出
5. 治理层：truth 分层、时间切分、评估/演进、memory profile、registry
6. 展示层：前端地图、状态带、证据链、审计面板、评估/演进视图

## 2. 关键设计
- 代理源显式标记：`proxy_source=true`
- 融合时代理上限：`proxy_weight_cap`
- 相关性惩罚：同源 agent 自动惩罚
- Readiness 与 hazard probability 分离
- 决策滞回：`issue_threshold`/`clear_threshold`
- 人工复核闸门：冲突高、过期、覆盖不足、代理依赖高 -> `manual_review`

## 3. 审计链
- 每次推理输出 `run_id`
- `runs/cycle_<run_id>.json` + `runs/audit/audit_<run_id>.json`
- SQLite `runs/registry.db` 记录 predict/eval/evolve 元数据
- 字段级 provenance：input_fields/upstream_sources/missing_fields/rule_version/model_version

## 4. 训练评估链
- truth label pipeline 产出 gold/silver/proxy 分层
- evaluate/evolve 强制时间切分 train/calibration/test
- 主结论只使用 gold/silver，proxy 只用于 auxiliary 分析
- 空间评估支持行政区级 + 网格级命中偏差
- nightly 批跑支持历史个例门禁与跨日趋势对比
- 评估报告支持 json/markdown/html
