#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERS_v3_4x5
==========

正式口径探索版：
- 使用 4 个 AI 领域 × 5 类伦理风险，不纳入“破坏生态环境”核心计算。
- ERM 基于第二轮词频中五类风险源结构的模糊推理结果。
- WoI 使用第六次讨论 v2.5 的统一 FAHP 问卷权重。
- CF 使用简化证据充分度修正：CF = sqrt(Q × C)。
- RC 不进入主公式，仅通过敏感性情景观察方法假设影响。

主公式：
    ERS = ERM × WoI × CF
    Q = min(EffectiveN / threshold, 1)
    C = 数据源加权覆盖度
    CF = sqrt(Q × C)

输出：
- ers_v3_results.csv
- ers_v3_component_breakdown.csv
- sensitivity_scenario_metrics.csv
- sensitivity_scenario_rankings.csv
- robustness_summary.csv
- ERS_v3_4x5_计算说明.md
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "数据集与说明" / "课题1-002四个重点领域人工智能伦理风险评估数据"
Q_ROOT = DATA_ROOT / "问卷-脱敏版"
FACTOR_WEIGHTS_CSV = ROOT / "第一次讨论" / "运行代码与结果" / "results" / "factor_priorities.csv"
VENDOR_DIR = ROOT / "第六次讨论" / "Priority_model_v1" / "vendor"
V25_SCRIPT = ROOT / "第六次讨论" / "v2.5_ERS" / "erm_calculator.py"

if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

if not V25_SCRIPT.exists():
    raise FileNotFoundError(f"找不到 v2.5 计算脚本: {V25_SCRIPT}")

spec = importlib.util.spec_from_file_location("ers_v25", V25_SCRIPT)
ers_v25 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["ers_v25"] = ers_v25
spec.loader.exec_module(ers_v25)


DOMAINS = ["个性化算法", "机器视觉", "自动驾驶", "服务机器人"]
RISKS_5 = [
    "决策不自主受控",
    "侵犯隐私",
    "加剧社会偏见或歧视",
    "权责归属不清或失当",
    "破坏社会公平",
]
RISK_SHORT = {
    "决策不自主受控": "自主性",
    "侵犯隐私": "隐私",
    "加剧社会偏见或歧视": "偏见歧视",
    "权责归属不清或失当": "责任",
    "破坏社会公平": "公平",
}

SOURCE_WEIGHT_SCENARIOS = {
    "baseline": {"WOS": 1.0, "知网": 0.9, "谷歌": 0.6, "百度": 0.5, "微博": 0.3},
    "equal": {"WOS": 1.0, "知网": 1.0, "谷歌": 1.0, "百度": 1.0, "微博": 1.0},
    "academic_high": {"WOS": 1.2, "知网": 1.1, "谷歌": 0.5, "百度": 0.4, "微博": 0.2},
    "media_high": {"WOS": 0.8, "知网": 0.7, "谷歌": 1.0, "百度": 0.9, "微博": 0.5},
    "weibo_low": {"WOS": 1.0, "知网": 0.9, "谷歌": 0.6, "百度": 0.5, "微博": 0.1},
}
THRESHOLDS = [60, 80, 100, 120]
CF_FORMS = ["sqrt", "product", "quantity_only", "coverage_only", "no_cf"]
WOI_FORMS = ["baseline", "flattened", "concentrated"]
ERM_FORMS = ["baseline", "weakened", "strengthened"]

BASELINE = {
    "source": "baseline",
    "threshold": 80,
    "cf_form": "sqrt",
    "woi_form": "baseline",
    "erm_form": "baseline",
}


@dataclass(frozen=True)
class Unit:
    domain: str
    risk: str

    @property
    def key(self) -> str:
        return f"{self.domain}-{self.risk}"


UNITS = [Unit(domain, risk) for domain in DOMAINS for risk in RISKS_5]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def set_source_weights(weights: dict[str, float]) -> None:
    ers_v25.SOURCE_WEIGHTS = weights.copy()


def build_erm(data, source_weights: dict[str, float]) -> dict[str, dict]:
    set_source_weights(source_weights)
    aggregation = ers_v25.compute_aggregation(data)
    factor_weights = ers_v25.load_factor_weights(str(FACTOR_WEIGHTS_CSV))
    raw = ers_v25.compute_all_erm(aggregation, factor_weights)
    return {f"{d}-{r}": value for (d, r), value in raw.items() if r in RISKS_5}


def build_cf(data, source_weights: dict[str, float], threshold: int, cf_form: str) -> dict[str, dict]:
    max_weight = max(source_weights.values())
    normalized = {k: v / max_weight for k, v in source_weights.items()}
    total_weight = sum(normalized.values())
    result: dict[str, dict] = {}
    for (domain, risk), drd in sorted(data.items()):
        if risk not in RISKS_5:
            continue
        effective_n = 0.0
        total_n = 0
        covered_weight = 0.0
        source_count = 0
        for source, source_data in drd.sources.items():
            n = int(source_data.total_articles)
            if n <= 0:
                continue
            q = normalized.get(source, 0.0)
            effective_n += n * (q**2)
            total_n += n
            covered_weight += q
            source_count += 1
        quantity = min(effective_n / threshold, 1.0) if threshold > 0 else 0.0
        coverage = min(covered_weight / total_weight, 1.0) if total_weight > 0 else 0.0
        if cf_form == "sqrt":
            cf = math.sqrt(max(quantity * coverage, 0.0))
        elif cf_form == "product":
            cf = quantity * coverage
        elif cf_form == "quantity_only":
            cf = quantity
        elif cf_form == "coverage_only":
            cf = coverage
        elif cf_form == "no_cf":
            cf = 1.0
        else:
            raise ValueError(f"未知 CF 形式: {cf_form}")
        result[f"{domain}-{risk}"] = {
            "CF": cf,
            "Q_数量充分度": quantity,
            "C_数据源加权覆盖度": coverage,
            "有效样本量_EffectiveN": effective_n,
            "总文章数": total_n,
            "数据源覆盖数": source_count,
            "CF阈值": threshold,
            "CF形式": cf_form,
        }
    return result


def load_woi() -> dict[str, dict]:
    raw = ers_v25.compute_woi_fahp_unified(str(Q_ROOT))
    return {f"{d}-{r}": value for (d, r), value in raw.items() if r in RISKS_5}


def transform_woi(base: dict[str, dict], form: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for domain in DOMAINS:
        keys = [f"{domain}-{risk}" for risk in RISKS_5]
        values = [base[key]["woi"] for key in keys]
        if form == "baseline":
            transformed = values
        elif form == "flattened":
            uniform = 1.0 / len(values)
            transformed = [0.5 * v + 0.5 * uniform for v in values]
        elif form == "concentrated":
            transformed = [v**1.25 for v in values]
        else:
            raise ValueError(f"未知 WoI 形式: {form}")
        total = sum(transformed)
        for key, value in zip(keys, transformed):
            result[key] = value / total if total else 1.0 / len(keys)
    return result


def transform_erm(base: dict[str, dict], form: str) -> dict[str, float]:
    values = [base[unit.key]["erm"] for unit in UNITS]
    avg = mean(values)
    result: dict[str, float] = {}
    for unit in UNITS:
        value = base[unit.key]["erm"]
        if form == "baseline":
            adjusted = value
        elif form == "weakened":
            adjusted = 0.5 * value + 0.5 * avg
        elif form == "strengthened":
            adjusted = ((value / avg) ** 1.25) * avg if avg else value
        else:
            raise ValueError(f"未知 ERM 形式: {form}")
        result[unit.key] = adjusted
    return result


def rank_scores(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {key: idx + 1 for idx, (key, _) in enumerate(ordered)}


def spearman(a: dict[str, int], b: dict[str, int]) -> float:
    keys = list(a)
    n = len(keys)
    if n < 2:
        return 1.0
    d2 = sum((a[k] - b[k]) ** 2 for k in keys)
    return 1 - 6 * d2 / (n * (n * n - 1))


def kendall_tau(a: dict[str, int], b: dict[str, int]) -> float:
    keys = list(a)
    concordant = 0
    discordant = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            sign = (a[ki] - a[kj]) * (b[ki] - b[kj])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def build_scores(
    erm_values: dict[str, float],
    woi_values: dict[str, float],
    cf_values: dict[str, dict],
) -> dict[str, float]:
    return {
        unit.key: erm_values[unit.key] * woi_values[unit.key] * cf_values[unit.key]["CF"]
        for unit in UNITS
    }


def classify(row: dict) -> str:
    if row["基准排名"] <= 5 and row["Top5频率"] >= 0.75:
        return "稳健高优先级"
    if row["基准排名"] <= 5 or row["Top5频率"] >= 0.40:
        return "条件高优先级"
    if row["中位排名"] <= 10:
        return "稳定中高优先级"
    if row["基准CF"] < 0.70:
        return "证据待补"
    return "常规监测"


def main() -> None:
    print("ERS_v3_4x5 计算开始")
    print(f"数据目录: {DATA_ROOT}")
    data = ers_v25.load_round2_data(str(DATA_ROOT))
    woi_base = load_woi()

    erm_cache = {
        name: build_erm(data, weights)
        for name, weights in SOURCE_WEIGHT_SCENARIOS.items()
    }
    cf_cache: dict[tuple[str, int, str], dict[str, dict]] = {}
    for source_name, source_weights in SOURCE_WEIGHT_SCENARIOS.items():
        for threshold in THRESHOLDS:
            for cf_form in CF_FORMS:
                cf_cache[(source_name, threshold, cf_form)] = build_cf(
                    data, source_weights, threshold, cf_form
                )

    baseline_erm = transform_erm(erm_cache[BASELINE["source"]], BASELINE["erm_form"])
    baseline_woi = transform_woi(woi_base, BASELINE["woi_form"])
    baseline_cf = cf_cache[(BASELINE["source"], BASELINE["threshold"], BASELINE["cf_form"])]
    baseline_scores = build_scores(baseline_erm, baseline_woi, baseline_cf)
    baseline_ranks = rank_scores(baseline_scores)
    baseline_top5 = {key for key, rank in baseline_ranks.items() if rank <= 5}

    baseline_max = max(baseline_scores.values())
    result_rows = []
    breakdown_rows = []
    for unit in sorted(UNITS, key=lambda u: baseline_ranks[u.key]):
        key = unit.key
        score = baseline_scores[key]
        row = {
            "领域": unit.domain,
            "风险维度": unit.risk,
            "风险简称": RISK_SHORT[unit.risk],
            "ERM": f"{baseline_erm[key]:.4f}",
            "WoI": f"{baseline_woi[key]:.6f}",
            "CF": f"{baseline_cf[key]['CF']:.6f}",
            "ERS_raw": f"{score:.6f}",
            "ERS_指数化": f"{(score / baseline_max * 100 if baseline_max else 0):.4f}",
            "全局排名": baseline_ranks[key],
            "Q_数量充分度": f"{baseline_cf[key]['Q_数量充分度']:.6f}",
            "C_数据源加权覆盖度": f"{baseline_cf[key]['C_数据源加权覆盖度']:.6f}",
            "有效样本量_EffectiveN": f"{baseline_cf[key]['有效样本量_EffectiveN']:.2f}",
            "总文章数": baseline_cf[key]["总文章数"],
            "数据源覆盖数": baseline_cf[key]["数据源覆盖数"],
            "WoI方法": woi_base[key].get("method", ""),
        }
        result_rows.append(row)
        detail = {
            **row,
            "Low激活": f"{erm_cache[BASELINE['source']][key]['activations']['Low']:.6f}",
            "Medium激活": f"{erm_cache[BASELINE['source']][key]['activations']['Medium']:.6f}",
            "High激活": f"{erm_cache[BASELINE['source']][key]['activations']['High']:.6f}",
            "数据处理占比": f"{erm_cache[BASELINE['source']][key]['factor_proportions'].get('数据处理', 0):.6f}",
            "算法开发占比": f"{erm_cache[BASELINE['source']][key]['factor_proportions'].get('算法开发', 0):.6f}",
            "产品开发占比": f"{erm_cache[BASELINE['source']][key]['factor_proportions'].get('产品开发', 0):.6f}",
            "部署使用占比": f"{erm_cache[BASELINE['source']][key]['factor_proportions'].get('部署使用', 0):.6f}",
            "外部制度规范占比": f"{erm_cache[BASELINE['source']][key]['factor_proportions'].get('外部制度规范', 0):.6f}",
        }
        breakdown_rows.append(detail)

    write_csv(OUT_DIR / "ers_v3_results.csv", result_rows)
    write_csv(OUT_DIR / "ers_v3_component_breakdown.csv", breakdown_rows)

    metric_rows = []
    ranking_rows = []
    scenario_id = 0
    for source_name in SOURCE_WEIGHT_SCENARIOS:
        for threshold in THRESHOLDS:
            for cf_form in CF_FORMS:
                for woi_form in WOI_FORMS:
                    for erm_form in ERM_FORMS:
                        scenario_id += 1
                        erm_values = transform_erm(erm_cache[source_name], erm_form)
                        woi_values = transform_woi(woi_base, woi_form)
                        cf_values = cf_cache[(source_name, threshold, cf_form)]
                        scores = build_scores(erm_values, woi_values, cf_values)
                        ranks = rank_scores(scores)
                        current_top5 = {key for key, rank in ranks.items() if rank <= 5}
                        rank_changes = [abs(ranks[key] - baseline_ranks[key]) for key in baseline_ranks]
                        scenario = {
                            "情景ID": f"S{scenario_id:04d}",
                            "数据源权重": source_name,
                            "CF阈值": threshold,
                            "CF形式": cf_form,
                            "WoI形式": woi_form,
                            "ERM形式": erm_form,
                            "Spearman": f"{spearman(baseline_ranks, ranks):.6f}",
                            "Kendall_Tau": f"{kendall_tau(baseline_ranks, ranks):.6f}",
                            "Top5重合率": f"{len(current_top5 & baseline_top5) / 5:.6f}",
                            "平均排名变化": f"{mean(rank_changes):.6f}",
                            "最大排名变化": max(rank_changes),
                        }
                        metric_rows.append(scenario)
                        for unit in UNITS:
                            key = unit.key
                            ranking_rows.append({
                                **scenario,
                                "领域": unit.domain,
                                "风险维度": unit.risk,
                                "领域-风险": key,
                                "ERS": f"{scores[key]:.6f}",
                                "情景排名": ranks[key],
                                "基准排名": baseline_ranks[key],
                                "绝对排名变化": abs(ranks[key] - baseline_ranks[key]),
                                "进入Top5": int(ranks[key] <= 5),
                            })

    write_csv(OUT_DIR / "sensitivity_scenario_metrics.csv", metric_rows)
    write_csv(OUT_DIR / "sensitivity_scenario_rankings.csv", ranking_rows)

    summary_rows = []
    by_key: dict[str, list[dict]] = {unit.key: [] for unit in UNITS}
    for row in ranking_rows:
        by_key[row["领域-风险"]].append(row)
    for unit in sorted(UNITS, key=lambda u: baseline_ranks[u.key]):
        rows = by_key[unit.key]
        ranks = [int(row["情景排名"]) for row in rows]
        top5_freq = sum(int(row["进入Top5"]) for row in rows) / len(rows)
        item = {
            "领域": unit.domain,
            "风险维度": unit.risk,
            "领域-风险": unit.key,
            "基准排名": baseline_ranks[unit.key],
            "基准ERS": f"{baseline_scores[unit.key]:.6f}",
            "基准CF": baseline_cf[unit.key]["CF"],
            "平均排名": f"{mean(ranks):.4f}",
            "中位排名": median(ranks),
            "最好排名": min(ranks),
            "最差排名": max(ranks),
            "Top5频率": top5_freq,
            "平均排名变化": f"{mean(abs(rank - baseline_ranks[unit.key]) for rank in ranks):.4f}",
        }
        item["稳健性分层"] = classify(item)
        item["基准CF"] = f"{item['基准CF']:.6f}"
        item["Top5频率"] = f"{item['Top5频率']:.6f}"
        summary_rows.append(item)

    write_csv(OUT_DIR / "robustness_summary.csv", summary_rows)

    top10 = result_rows[:10]
    metrics_float = [
        {
            "Spearman": float(row["Spearman"]),
            "Kendall_Tau": float(row["Kendall_Tau"]),
            "Top5重合率": float(row["Top5重合率"]),
            "平均排名变化": float(row["平均排名变化"]),
            "最大排名变化": int(row["最大排名变化"]),
        }
        for row in metric_rows
    ]
    report = [
        "# ERS_v3_4x5 计算说明",
        "",
        "## 1. 计算口径",
        "",
        "本版采用 `4个AI领域 × 5类伦理风险`，不将“破坏生态环境”纳入核心计算。",
        "",
        "主公式：",
        "",
        "```text",
        "ERS = ERM × WoI × CF",
        "Q = min(EffectiveN / threshold, 1)",
        "C = 数据源加权覆盖度",
        "CF = sqrt(Q × C)",
        "```",
        "",
        "其中，ERM 来自第二轮词频中的五类风险源结构；WoI 来自专家和公众问卷的统一FAHP权重；CF 是简化证据充分度修正。RC 不进入主公式。",
        "",
        "## 2. 基准结果 Top 10",
        "",
        "| 排名 | 领域 | 风险维度 | ERM | WoI | CF | ERS指数化 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in top10:
        report.append(
            f"| {row['全局排名']} | {row['领域']} | {row['风险维度']} | "
            f"{float(row['ERM']):.2f} | {float(row['WoI']):.4f} | "
            f"{float(row['CF']):.4f} | {float(row['ERS_指数化']):.2f} |"
        )
    report.extend([
        "",
        "## 3. 敏感性检验设计",
        "",
        f"本版共构造 `{len(metric_rows)}` 个情景，扰动数据源权重、CF阈值、CF形式、WoI形式和ERM形式。",
        "",
        f"- Spearman 最小值：{min(x['Spearman'] for x in metrics_float):.3f}，中位数：{median(x['Spearman'] for x in metrics_float):.3f}",
        f"- Kendall Tau 最小值：{min(x['Kendall_Tau'] for x in metrics_float):.3f}，中位数：{median(x['Kendall_Tau'] for x in metrics_float):.3f}",
        f"- Top5重合率最小值：{min(x['Top5重合率'] for x in metrics_float):.1%}，中位数：{median(x['Top5重合率'] for x in metrics_float):.1%}",
        f"- 情景平均排名变化中位数：{median(x['平均排名变化'] for x in metrics_float):.2f}",
        f"- 单情景最大排名变化最大值：{max(x['最大排名变化'] for x in metrics_float)}",
        "",
        "## 4. 文件说明",
        "",
        "- `ers_v3_results.csv`：基准ERS结果。",
        "- `ers_v3_component_breakdown.csv`：ERM、WoI、CF及风险源结构分解。",
        "- `sensitivity_scenario_metrics.csv`：各敏感性情景的整体排序稳定性。",
        "- `sensitivity_scenario_rankings.csv`：各情景下20项领域-风险组合排名明细。",
        "- `robustness_summary.csv`：每个领域-风险组合的稳健性分层。",
        "",
        "## 5. 解释边界",
        "",
        "ERS 是 AI伦理风险关注优先级指数，不是企业真实风险分数、事故概率、损失金额或保险费率。",
    ])
    (OUT_DIR / "ERS_v3_4x5_计算说明.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("ERS_v3_4x5 计算完成")
    print(f"基准结果: {OUT_DIR / 'ers_v3_results.csv'}")
    print(f"稳健性汇总: {OUT_DIR / 'robustness_summary.csv'}")
    print(f"敏感性情景数: {len(metric_rows)}")


if __name__ == "__main__":
    main()
