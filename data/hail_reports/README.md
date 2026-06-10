# 冰雹报告文件格式

将业务冰雹报告放在本目录的 `*.csv` 文件中，评估与演进会自动读取并并入真值标签。

## 字段要求

推荐列（至少包含 `time`）：

- `time` 或 `timestamp` 或 `datetime`：时间（ISO 或 `YYYY-MM-DD HH:MM[:SS]`）
- `city`：城市名（可选，用于多城市混合文件过滤）
- `hail`：是否冰雹（`1/0`，可选；留空默认按 1 处理）
- `diameter_mm`：雹径（可选）
- `source`：来源（可选）

## 示例

```csv
time,city,hail,diameter_mm,source
2025-03-01 14:20:00,Tianjin,1,8,manual_report
2025-03-01 14:55:00,Tianjin,1,12,manual_report
2025-03-02 03:10:00,Tianjin,1,6,manual_report
```
