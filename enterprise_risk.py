"""Deterministic enterprise AI-ethics risk profiling (paper chapters 4–5)."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Mapping

from data_loader import DATA_DIR, DOMAINS, ERS_RESULTS_CSV
from enterprise_risk_benchmarks import (
    ENTERPRISE_ERS_THRESHOLD_MEAN,
    EXPOSURE_THRESHOLD_Q90,
    INDUSTRY_AVG_PATENT_DENSITY,
    INDUSTRY_AVG_WORD_FREQUENCY_INTENSITY,
    MAX_WORD_EXPOSURE,
)


PATENT_EXPOSURE_WEIGHT = 0.3
WORD_EXPOSURE_WEIGHT = 0.7
PATENT_ERS_WEIGHT = 0.7
WORD_ERS_WEIGHT = 0.3

FOCUS_DOMAINS = tuple(DOMAINS)
CHINESE_FONT_PATH = Path(__file__).resolve().parent / "assets" / "NotoSansSC-VF.ttf"


class EnterpriseRiskValidationError(ValueError):
    """Raised when the enterprise profile input cannot be evaluated."""


QUADRANT_INFO = {
    "Q1": {
        "label": "Q1 高暴露—高 ERS",
        "explanation": "企业 AI 应用暴露程度较高，同时所布局技术领域的伦理风险严重程度也较高，应优先采取强风险控制或退出措施。",
    },
    "Q2": {
        "label": "Q2 低暴露—高 ERS",
        "explanation": "企业当前 AI 暴露程度较低，但所布局技术领域本身具有较高伦理风险，应重点防范低频高损风险。",
    },
    "Q3": {
        "label": "Q3 低暴露—低 ERS",
        "explanation": "企业当前 AI 暴露程度和风险严重程度均相对较低，可在设定风险容忍度的基础上采取风险自留和持续监测。",
    },
    "Q4": {
        "label": "Q4 高暴露—低 ERS",
        "explanation": "企业 AI 应用暴露较高，但风险严重程度相对较低，应重点通过损失控制和风险分散降低高频风险累积。",
    },
}

# 第五章的确定性映射。这里不调用大模型，也不允许由模型选择主要策略。
QUADRANT_STRATEGIES = {
    "Q1": [
        {
            "name": "风险撤出",
            "description": "对于高频高损、无法通过其他控制措施降低到可接受范围，或者存在不可逆损害的风险，应减少或终止相关风险暴露。",
        }
    ],
    "Q2": [
        {
            "name": "风险对冲",
            "description": "通过 ESG、治理、声誉等缓冲能力降低风险事件的衍生冲击。",
        },
        {
            "name": "保险转移",
            "description": "通过保险或其他风险融资手段转移低概率、高损失风险。",
        },
    ],
    "Q3": [
        {
            "name": "风险自留",
            "description": "在风险可承受的情况下，设置风险容忍度、自留限额和伦理风险储备，并持续监测风险变化。",
        }
    ],
    "Q4": [
        {
            "name": "损失控制",
            "description": "通过数据最小化、公平性检测、可解释性工具、人工复核、模型测试等措施降低风险发生频率。",
        },
        {
            "name": "风险分散",
            "description": "通过多数据源、多模型、多供应商等方式降低风险集中度。",
        },
        {
            "name": "风险自留",
            "description": "对剩余的低损风险设定可承受范围并持续监测。",
        },
    ],
}

MANAGEMENT_STRATEGY_NAMES = frozenset(
    strategy["name"]
    for strategies in QUADRANT_STRATEGIES.values()
    for strategy in strategies
)


def _non_negative(value: float, field_name: str) -> float:
    value = float(value)
    if value < 0:
        raise EnterpriseRiskValidationError(f"{field_name}不能小于 0。")
    return value


def calculate_patent_density(ai_patents: float, total_patents: float) -> float:
    """企业 AI 专利密度 = 重点领域 AI 专利总数 / 企业专利总数。"""
    ai_patents = _non_negative(ai_patents, "重点领域 AI 专利总数")
    total_patents = _non_negative(total_patents, "企业专利总数")
    return ai_patents / total_patents if total_patents else 0.0


def calculate_patent_exposure(patent_density: float) -> float:
    """专利暴露倍数 = 企业 AI 专利密度 / 行业平均 AI 专利密度。"""
    if INDUSTRY_AVG_PATENT_DENSITY <= 0:
        raise EnterpriseRiskValidationError("行业平均 AI 专利密度无效，无法计算专利暴露倍数。")
    return float(patent_density) / INDUSTRY_AVG_PATENT_DENSITY


def calculate_word_frequency_intensity(ai_word_frequency: float, annual_report_words: float) -> float:
    """企业 AI 词频强度 = 年报 AI 总词频 / 年报总词数。"""
    ai_word_frequency = _non_negative(ai_word_frequency, "年报 AI 关键词总词频")
    annual_report_words = _non_negative(annual_report_words, "企业年报总词数")
    if annual_report_words <= 0:
        raise EnterpriseRiskValidationError("企业年报总词数必须大于 0。")
    return ai_word_frequency / annual_report_words


def calculate_word_exposure(word_frequency_intensity: float) -> float:
    """词频暴露倍数 = 企业 AI 词频强度 / 行业平均 AI 词频强度。"""
    if INDUSTRY_AVG_WORD_FREQUENCY_INTENSITY <= 0:
        raise EnterpriseRiskValidationError("行业平均 AI 词频强度无效，无法计算词频暴露倍数。")
    return float(word_frequency_intensity) / INDUSTRY_AVG_WORD_FREQUENCY_INTENSITY


def calculate_exposure_multiple(patent_exposure: float, word_exposure: float) -> float:
    """暴露倍数 = 0.3 × 专利暴露倍数 + 0.7 × 词频暴露倍数。"""
    return PATENT_EXPOSURE_WEIGHT * float(patent_exposure) + WORD_EXPOSURE_WEIGHT * float(word_exposure)


def load_ers_statistics() -> tuple[dict[str, float], float]:
    """Read existing 20 ERS ground-truth rows; do not duplicate ERS values."""
    path = Path(DATA_DIR) / ERS_RESULTS_CSV
    if not path.exists():
        raise EnterpriseRiskValidationError(f"未找到现有 ERS 数据文件：{path}")

    values_by_domain: dict[str, list[float]] = {domain: [] for domain in FOCUS_DOMAINS}
    all_values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            domain = row.get("领域")
            if domain not in values_by_domain:
                continue
            try:
                value = float(row["ERS_指数化"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EnterpriseRiskValidationError("ERS 原始数据存在无法读取的分数。") from exc
            values_by_domain[domain].append(value)
            all_values.append(value)

    if len(all_values) != len(FOCUS_DOMAINS) * 5 or any(not values for values in values_by_domain.values()):
        raise EnterpriseRiskValidationError("ERS 原始数据不完整，无法计算领域 ERS 均值。")
    return (
        {domain: sum(values) / len(values) for domain, values in values_by_domain.items()},
        sum(all_values) / len(all_values),
    )


def calculate_patent_ers(domain_patents: Mapping[str, float]) -> float:
    """AI 专利 ERS = Σ(领域专利占比 × 该领域 ERS 均值)。"""
    counts = {domain: _non_negative(domain_patents.get(domain, 0), f"{domain}相关专利数量") for domain in FOCUS_DOMAINS}
    total_ai_patents = sum(counts.values())
    if total_ai_patents == 0:
        return 0.0
    domain_means, _ = load_ers_statistics()
    return sum((counts[domain] / total_ai_patents) * domain_means[domain] for domain in FOCUS_DOMAINS)


def calculate_word_ers(word_exposure: float) -> float:
    """AI 词频 ERS = ln(1 + 词频暴露) / ln(1 + 样本最大值) × 行业平均 ERS。"""
    if MAX_WORD_EXPOSURE <= 0:
        raise EnterpriseRiskValidationError("样本最大词频暴露倍数无效，无法计算 AI 词频 ERS。")
    _, industry_avg_ers = load_ers_statistics()
    return (math.log1p(max(0.0, float(word_exposure))) / math.log1p(MAX_WORD_EXPOSURE)) * industry_avg_ers


def calculate_enterprise_ers(patent_ers: float, word_ers: float) -> float:
    """企业综合 ERS = 0.7 × AI 专利 ERS + 0.3 × AI 词频 ERS。"""
    return PATENT_ERS_WEIGHT * float(patent_ers) + WORD_ERS_WEIGHT * float(word_ers)


def classify_quadrant(exposure_multiple: float, enterprise_ers: float) -> str:
    """Classify with the paper's q90 exposure threshold and mean ERS threshold."""
    high_exposure = float(exposure_multiple) >= EXPOSURE_THRESHOLD_Q90
    high_ers = float(enterprise_ers) >= ENTERPRISE_ERS_THRESHOLD_MEAN
    if high_exposure and high_ers:
        return "Q1"
    if not high_exposure and high_ers:
        return "Q2"
    if not high_exposure and not high_ers:
        return "Q3"
    return "Q4"


def evaluate_enterprise_risk(
    *,
    total_patents: float,
    personalized_algorithm_patents: float,
    machine_vision_patents: float,
    autonomous_driving_patents: float,
    service_robot_patents: float,
    annual_report_words: float,
    ai_word_frequency: float,
) -> dict:
    """Calculate the full deterministic enterprise profile for Streamlit state."""
    total_patents = _non_negative(total_patents, "企业专利总数")
    annual_report_words = _non_negative(annual_report_words, "企业年报总词数")
    if annual_report_words <= 0:
        raise EnterpriseRiskValidationError("企业年报总词数必须大于 0。")

    domain_patents = {
        "个性化算法": _non_negative(personalized_algorithm_patents, "个性化算法相关专利数量"),
        "机器视觉": _non_negative(machine_vision_patents, "机器视觉相关专利数量"),
        "自动驾驶": _non_negative(autonomous_driving_patents, "自动驾驶相关专利数量"),
        "服务机器人": _non_negative(service_robot_patents, "服务机器人相关专利数量"),
    }
    total_ai_patents = sum(domain_patents.values())
    if total_ai_patents > total_patents:
        raise EnterpriseRiskValidationError("四个 AI 领域专利数量之和不能大于企业专利总数。")

    patent_density = calculate_patent_density(total_ai_patents, total_patents)
    patent_exposure = calculate_patent_exposure(patent_density)
    word_frequency_intensity = calculate_word_frequency_intensity(ai_word_frequency, annual_report_words)
    word_exposure = calculate_word_exposure(word_frequency_intensity)
    exposure_multiple = calculate_exposure_multiple(patent_exposure, word_exposure)
    patent_ers = calculate_patent_ers(domain_patents)
    word_ers = calculate_word_ers(word_exposure)
    enterprise_ers = calculate_enterprise_ers(patent_ers, word_ers)
    quadrant = classify_quadrant(exposure_multiple, enterprise_ers)

    return {
        "patent_density": patent_density,
        "word_frequency_intensity": word_frequency_intensity,
        "patent_exposure": patent_exposure,
        "word_exposure": word_exposure,
        "exposure_multiple": exposure_multiple,
        "patent_ers": patent_ers,
        "word_ers": word_ers,
        "enterprise_ers": enterprise_ers,
        "exposure_threshold": EXPOSURE_THRESHOLD_Q90,
        "ers_threshold": ENTERPRISE_ERS_THRESHOLD_MEAN,
        "quadrant": quadrant,
        "quadrant_label": QUADRANT_INFO[quadrant]["label"],
        "explanation": QUADRANT_INFO[quadrant]["explanation"],
        "strategies": [dict(strategy) for strategy in QUADRANT_STRATEGIES[quadrant]],
    }


def profile_prompt_context(profile: Mapping[str, object] | None) -> str:
    """Build bounded context for the advice model from a locked profile only."""
    if not profile:
        return "未完成企业风险画像；不适用企业象限策略约束。"
    strategies = profile.get("strategies", [])
    names = [item.get("name", "") for item in strategies if isinstance(item, Mapping)]
    return (
        "企业风险评估结果（由 Python 根据第四、五章公式锁定，不得重新判断）：\n"
        f"- 综合暴露倍数：{float(profile.get('exposure_multiple', 0.0)):.2f}\n"
        f"- 企业综合 ERS：{float(profile.get('enterprise_ers', 0.0)):.2f}\n"
        f"- 所属象限：{profile.get('quadrant_label', profile.get('quadrant', ''))}\n"
        f"- 本企业优先适用的正式风险管理策略：{'、'.join(names)}"
    )


def profile_strategy_names(profile: Mapping[str, object] | None) -> set[str]:
    """Return the deterministic primary strategies allowed for advice output."""
    if not profile:
        return set(MANAGEMENT_STRATEGY_NAMES)
    strategies = profile.get("strategies", [])
    return {
        str(item.get("name"))
        for item in strategies
        if isinstance(item, Mapping) and item.get("name") in MANAGEMENT_STRATEGY_NAMES
    }


def create_quadrant_figure(profile: Mapping[str, object]):
    """Show only the current enterprise's quadrant and its coordinates."""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    # Streamlit Cloud's Linux image does not include Windows Chinese fonts.
    # Bundle an OFL Noto Sans SC font with the project and load it by path so
    # the generated figure is identical locally and in cloud deployments.
    chinese_font = FontProperties(fname=str(CHINESE_FONT_PATH)) if CHINESE_FONT_PATH.exists() else None
    plt.rcParams['axes.unicode_minus'] = False

    exposure = float(profile["exposure_multiple"])
    enterprise_ers = float(profile["enterprise_ers"])
    quadrant = str(profile["quadrant"])
    label = str(profile.get("quadrant_label", quadrant))

    # The figure is intentionally cropped to the enterprise's own quadrant;
    # it is not a full four-quadrant comparison chart.
    if quadrant == "Q1":
        x_min, x_max = EXPOSURE_THRESHOLD_Q90, max(exposure * 1.15, EXPOSURE_THRESHOLD_Q90 * 1.10)
        y_min, y_max = ENTERPRISE_ERS_THRESHOLD_MEAN, max(enterprise_ers * 1.15, ENTERPRISE_ERS_THRESHOLD_MEAN * 1.10)
    elif quadrant == "Q2":
        x_min, x_max = 0.0, EXPOSURE_THRESHOLD_Q90
        y_min, y_max = ENTERPRISE_ERS_THRESHOLD_MEAN, max(enterprise_ers * 1.15, ENTERPRISE_ERS_THRESHOLD_MEAN * 1.10)
    elif quadrant == "Q3":
        x_min, x_max = 0.0, EXPOSURE_THRESHOLD_Q90
        y_min, y_max = 0.0, ENTERPRISE_ERS_THRESHOLD_MEAN
    elif quadrant == "Q4":
        x_min, x_max = EXPOSURE_THRESHOLD_Q90, max(exposure * 1.15, EXPOSURE_THRESHOLD_Q90 * 1.10)
        y_min, y_max = 0.0, ENTERPRISE_ERS_THRESHOLD_MEAN
    else:
        raise EnterpriseRiskValidationError(f"无法识别企业风险象限：{quadrant}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.scatter([exposure], [enterprise_ers], color="#c0392b", s=85, zorder=3)
    ax.annotate(
        f"当前企业\n({exposure:.2f}, {enterprise_ers:.2f})",
        (exposure, enterprise_ers),
        xytext=(8, 8),
        textcoords="offset points",
        fontproperties=chinese_font,
    )
    ax.text(
        0.5,
        0.93,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontproperties=chinese_font,
    )
    ax.set_xlabel("暴露倍数", fontproperties=chinese_font)
    ax.set_ylabel("企业综合 ERS", fontproperties=chinese_font)
    ax.set_title("企业 AI 伦理风险画像", fontproperties=chinese_font)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig
