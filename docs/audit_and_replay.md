# Audit And Replay

## 1. 审计产物
- 推理产物：`runs/cycle_<run_id>.json`
- 审计产物：`runs/audit/audit_<run_id>.json`
- 注册表：`runs/registry.db`

## 2. 审计字段
- 输入来源与健康状态
- 每个 EvidenceCard 的 provenance
- 融合贡献（per-agent contribution）
- 决策链（issue/clear/人工复核闸门触发原因）

## 3. API
- `GET /api/v1/audit/{run_id}`：查看完整审计记录
- `POST /api/v1/replay/case`：回放指定 run
- `POST /api/v1/replay/compare`：比较两个 run 的差异
- `GET /api/v1/registry/eval/{run_id}`：查看评估 run 元数据
- `GET /api/v1/registry/evolve/{run_id}`：查看演进 run 元数据

## 4. 审计建议
- 在业务使用中把 run_id 与值班记录绑定
- 关键事件要求导出 audit + registry 记录以满足复盘
