from __future__ import annotations

from weather_agent.models import Observation


def build_explicit_plan(obs: Observation) -> list[str]:
    plan = [
        "Step 1: 拉取多源数据并完成时空对齐（模式/雷达/客观概率）。",
        "Step 2: 计算数据质量评分与跨源一致性，识别低可信信号。",
        "Step 3: 执行环境诊断（CAPE/DCAPE/切变/WBZ/月度阈值）。",
        "Step 4: 执行雷达与短临外推，判断未来0-2小时演变。",
        "Step 5: 融合证据并输出冲突类型与系统置信度。",
        "Step 6: 应用概率校准器（若存在）得到后验概率。",
        "Step 7: 生成预警建议并写入回放，用于外环自主演进。",
    ]

    if obs.source_meta.get("mode") == "live":
        plan.append("Step 8: 调用长期记忆参数（权重+校准器）进行在线自适应。")
    return plan
