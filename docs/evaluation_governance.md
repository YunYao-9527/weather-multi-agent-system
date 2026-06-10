# Evaluation Governance

## 1. 标签治理
- 标签分层：`gold` / `silver` / `proxy`
- Headline 评估与训练：仅允许 `gold/silver`
- `proxy`：仅用于 auxiliary 分析，不能作为正式结论

## 2. 数据泄漏防护
- 强制时间切分：train -> calibration -> test（按时间顺序）
- baseline 与 enhanced 使用同一 test 集
- 输出 split manifest（counts/index/windows）

## 3. 指标体系
- 点级：precision / recall / f1 / brier
- 事件级：POD / FAR / CSI
- 时效：lead time
- 稳定性：等级抖动率、升级/降级一致性
- 空间指标：行政区 `admin_hit_rate/admin_coverage_bias` + 网格级 `grid_hit_rate/grid_coverage_bias/grid_csi`
- 分层评估：按灾种、按数据质量切片
- 不确定性：bootstrap 置信区间

## 4. 报告
- 评估输出三种格式：JSON / Markdown / HTML
- 所有报告必须包含：时间窗、样本数、标签分层、覆盖率、改进幅度与置信区间
- 不再使用固定“10%达标”二值判定，改为持续跟踪多指标改进幅度

## 5. 运行要求
- 若 `truth_policy=require` 且 `qualified_coverage_ratio < min_truth_coverage`，评估失败
- 若 `truth_policy=require` 且 `total_positive_labels < min_total_positive_labels`，评估失败
- 若真值覆盖不足，不允许“伪提升结论”进入 headline
- 若测试窗口无正例，F1/POD/CSI 等事件指标不具统计判别力，必须先换窗口再评估
- 建议先执行窗口扫描（`/api/v1/windows/scan` 或 `scripts/find_positive_windows.ps1`）再跑评估/演进
