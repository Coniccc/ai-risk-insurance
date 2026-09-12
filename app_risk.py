import html
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="AI 伦理风险识别与管理建议", page_icon="🛡️", layout="wide")

PAGE_SIZE = 20

BASE_DIR = Path(__file__).resolve().parent
BG_IMAGE = BASE_DIR / "background.jpg"

# 习近平总书记四个「时代之问」
QUOTE = (
    "当机器开始思考，人类如何与之相处？当算法参与决策，安全如何保障？\n"
    "当技术挑战伦理，治理如何跟上？当鸿沟不断拉大，普惠如何实现？"
)
QUOTE_ATTRIBUTION = "——习近平总书记在 2026 世界人工智能大会提出的四个「时代之问」"


# ---------------------------------------------------------------------------
# API Key 处理
# ---------------------------------------------------------------------------
def _ensure_api_key() -> str:
    """获取 DashScope API Key：本地读环境变量，云端读 st.secrets 并注入环境变量。"""
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if key:
        return key
    try:
        key = st.secrets.get("DASHSCOPE_API_KEY", "") or ""
    except Exception:
        key = ""
    if key:
        os.environ["DASHSCOPE_API_KEY"] = key
    return key


API_KEY = _ensure_api_key()

from data_loader import (
    DOMAINS,
    ERS_DESCRIPTION,
    GENERIC_DOMAIN,
    load_data_if_needed,
)
from data_source import load_news_df, load_policy_df
from enterprise_risk import (
    EnterpriseRiskValidationError,
    create_quadrant_figure,
    evaluate_enterprise_risk,
)
from file_parser import parse_uploaded_file
from risk_service import RiskService

# 领域选择列表（4 个重点领域 + 其他通用领域）
DOMAIN_OPTIONS = DOMAINS + [GENERIC_DOMAIN]


# ---------------------------------------------------------------------------
# 渲染工具
# ---------------------------------------------------------------------------
def _md_escape(s: str) -> str:
    return html.escape(str(s))


# 后端仍保存 Pydantic 结构化对象；以下函数仅负责稳定、可读的前端呈现，
# 不会重新判断风险或修改任何结论字段。
RISK_TYPE_LABELS = {
    "autonomy": "决策不自主受控",
    "privacy": "侵犯隐私",
    "bias": "加剧社会偏见或歧视",
    "fairness": "破坏社会公平",
    "accountability": "权责归属不清或失当",
}
RISK_STATUS_LABELS = {
    "identified": "已识别风险",
    "verify": "潜在风险/待核验风险",
    "no_obvious_risk": "当前未发现明显风险",
}
RISK_STATUS_DISPLAY_ORDER = {
    "identified": 0,
    "verify": 1,
    "no_obvious_risk": 2,
}


def _format_report_list(items, empty_text: str, indent: int = 0) -> str:
    """Format structured list fields as Markdown report bullets."""
    prefix = " " * indent + "- "
    if not items:
        return f"{prefix}{_md_escape(empty_text)}"
    return "\n".join(f"{prefix}{_md_escape(item)}" for item in items)


def _display_risk_order(risk_result):
    """Return a presentation-only ordering without mutating locked results."""
    def sort_key(risk):
        status_order = RISK_STATUS_DISPLAY_ORDER[risk.status]
        # Only identified risks carry a backend-bound ERS score.  Missing or
        # malformed display values sort after numeric scores without changing
        # the underlying result.
        try:
            score_order = -float(risk.ers_score) if risk.ers_score is not None else float("inf")
        except (TypeError, ValueError):
            score_order = float("inf")
        return status_order, score_order

    return sorted(risk_result.risks, key=sort_key)


def _format_risk_result_for_display(risk_result) -> str:
    lines = ["根据已校验的企业场景信息与参考资料，识别出以下 AI 伦理风险：", "", "------", ""]
    for index, risk in enumerate(_display_risk_order(risk_result), start=1):
        type_label = RISK_TYPE_LABELS[risk.risk_type]
        status_label = RISK_STATUS_LABELS[risk.status]
        lines.extend(
            [
                f"### {index}. 【{status_label}】{type_label}",
                "",
                "- **具体表现**：",
                _format_report_list(
                    risk.evidence_from_input,
                    "输入中未提供该风险的直接证据。",
                    indent=2,
                ),
                f"- **成因/判断依据**：{_md_escape(risk.rationale)}",
            ]
        )
        if risk.status == "verify":
            lines.extend(
                [
                    "- **待核验事项**：",
                    _format_report_list(
                        risk.verification_needed,
                        "待补充关键核验信息。",
                        indent=2,
                    ),
                ]
            )
        if risk.ers_score is not None:
            lines.append(f"- **ERS 分数：{_md_escape(risk.ers_score)}**")
        lines.extend(["", "------", ""])
    return "\n".join(lines)


def _format_advice_result_for_display(advice_result, risk_result) -> str:
    lines = ["基于已锁定的风险识别结论，现提出以下必要、直接、可执行的风险管理建议：", "", "------", ""]
    advice_by_type = {advice.risk_type: advice for advice in advice_result.advice}
    ordered_risks = _display_risk_order(risk_result)
    for index, risk in enumerate(ordered_risks, start=1):
        advice = advice_by_type[risk.risk_type]
        type_label = RISK_TYPE_LABELS[advice.risk_type]
        status_label = RISK_STATUS_LABELS[advice.status]
        score_suffix = f"（ERS: {_md_escape(advice.ers_score)}）" if advice.ers_score else ""
        lines.extend(
            [
                f"### **{index}. 针对【{status_label}】{type_label}{score_suffix}**",
                "",
                f"**核心问题**：{_md_escape(risk.rationale)}",
                "",
                "**风险管理建议**：",
            ]
        )
        if advice.recommendations:
            lines.extend(
                f"- **【{_md_escape(item.strategy)}】** {_md_escape(item.recommendation)}"
                for item in advice.recommendations
            )
        else:
            lines.append("- 建议持续维护并监测现有控制。")
        lines.append("")
        # if risk.existing_controls:
        #     controls = "；".join(_md_escape(item) for item in risk.existing_controls)
        #     lines.append(f"> 注：已识别的相关控制包括：{controls}。建议优先维护、核验或改进，避免重复建设。")
        # else:
        #     lines.append("> 注：输入未说明相关控制情况，不代表不存在；建议先核实现有安排后再补充控制措施。")
        lines.extend(["", "------", ""])
    return "\n".join(lines)


def _render_link_list(items: list[dict], link_key: str) -> None:
    """以带超链接的列表渲染若干条记录。"""
    lis = []
    for it in items:
        title = _md_escape(it["标题"])
        source = it.get("来源") or it.get("发布机构") or ""
        meta_bits = [b for b in [it.get("发布时间"), source] if b]
        meta = " · ".join(meta_bits)
        link = it.get(link_key, "")
        parts = [f"<li><strong>{title}</strong>"]
        if meta:
            parts.append(f'<br/><span style="color:#888">{_md_escape(meta)}</span>')
        if link:
            parts.append(
                f'<br/><a href="{_md_escape(link)}" target="_blank" rel="noopener noreferrer">'
                "查看原文 ↗</a>"
            )
        parts.append("</li>")
        lis.append("".join(parts))
    st.markdown("<ul>" + "".join(lis) + "</ul>", unsafe_allow_html=True)


def _pagination(total_items: int, key: str) -> tuple[int, int]:
    """渲染「上一页 / 下一页」分页控件，返回 (start, size)。"""
    total_pages = max(1, (total_items + PAGE_SIZE - 1) // PAGE_SIZE)
    page_key = f"{key}_page"
    st.session_state.setdefault(page_key, 1)

    # 读取并夹紧页码，防止筛选后总数变小时越界
    page = min(max(1, int(st.session_state[page_key])), total_pages)
    st.session_state[page_key] = page

    # 回调在脚本重跑之前执行，因此重跑时 page / disabled 都是最新状态
    def _go_prev():
        st.session_state[page_key] = max(1, st.session_state[page_key] - 1)

    def _go_next():
        st.session_state[page_key] = min(total_pages, st.session_state[page_key] + 1)

    c1, c2, c3 = st.columns([1, 1, 3])
    c1.button(
        "⬅ 上一页",
        disabled=(page <= 1),
        key=f"{key}_prev",
        width="stretch",
        on_click=_go_prev,
    )
    c2.button(
        "下一页 ➡",
        disabled=(page >= total_pages),
        key=f"{key}_next",
        width="stretch",
        on_click=_go_next,
    )
    c3.markdown(f"第 **{page}** / {total_pages} 页 · 共 {total_items} 条")

    start = (page - 1) * PAGE_SIZE
    return start, PAGE_SIZE


def _filter_df(df, kw: str, columns: list[str]):
    """按关键词在指定列中做不区分大小写的包含匹配。"""
    kw = (kw or "").strip()
    if not kw:
        return df
    mask = None
    for col in columns:
        m = df[col].astype(str).str.contains(kw, case=False, na=False)
        mask = m if mask is None else (mask | m)
    return df[mask] if mask is not None else df


# ---------------------------------------------------------------------------
# 知识库（风险识别用）
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="正在载入知识库…")
def _init_knowledge_base():
    if not API_KEY:
        return False, "未配置 DASHSCOPE_API_KEY"
    try:
        _, msg = load_data_if_needed()
        return True, msg
    except Exception as exc:  # noqa: BLE001
        return False, f"知识库载入失败：{exc}"


def _api_key_error_message() -> str:
    return (
        "⚠️ 未配置 DASHSCOPE_API_KEY。\n\n"
        "（政策栏 / 资讯栏不依赖 API Key，可正常使用。）"
    )


# ---------------------------------------------------------------------------
# 标签页 1：风险识别与建议
# ---------------------------------------------------------------------------
def render_risk_tab():
    ok, load_msg = _init_knowledge_base()
    if not ok:
        st.error(_api_key_error_message())
        return

    if "risk_service" not in st.session_state:
        st.session_state["risk_service"] = RiskService()
    if "identified_risks" not in st.session_state:
        st.session_state["identified_risks"] = None

    # 领域选择
    domain = st.radio(
        "请选择 AI 重点领域（重点领域将结合 ERS 指数分析，其他通用领域直接给出潜在风险）",
        options=DOMAIN_OPTIONS,
        horizontal=True,
        key="domain_selector",
    )
    is_focus = domain != GENERIC_DOMAIN

    user_text = st.text_area(
        "请输入企业条款 / 专利 / 项目内容",
        height=200,
        placeholder="例如：本公司将使用人工智能算法对海量用户数据进行自动化画像与评分……",
    )

    # 文件上传：解析文件内容作为识别/建议的输入来源
    uploaded_file = st.file_uploader(
        "可上传项目文件作为补充（支持 txt / md / pdf / docx / xlsx / xls / csv 等）",
        type=["txt", "md", "pdf", "docx", "xlsx", "xls", "csv", "json", "log", "py", "yaml", "yml", "html", "htm"],
        key="risk_uploader",
    )

    file_text = ""
    if uploaded_file is not None:
        try:
            file_text = parse_uploaded_file(uploaded_file.name, uploaded_file.getvalue())
        except Exception as exc:  # noqa: BLE001
            st.error(f"文件解析失败：{exc}")
        else:
            st.caption(f"已解析文件「{uploaded_file.name}」，共 {len(file_text)} 字符")

    col_identify, col_advice = st.columns(2)
    with col_identify:
        click_identify = st.button("🔍 识别 AI 伦理风险", type="primary", width="stretch")
    with col_advice:
        click_advice = st.button("📋 生成风险管理建议", width="stretch")

    # 合并文本输入与文件内容，作为最终分析对象
    combined_text = "\n\n".join(
        p for p in [user_text.strip(), file_text.strip()] if p
    )

    if click_identify:
        if not combined_text:
            st.warning("请先输入内容或上传文件。")
        else:
            with st.spinner("正在识别 AI 伦理风险…"):
                result = st.session_state["risk_service"].identify_risks(combined_text, domain)
            st.session_state["identified_risks"] = result
            st.session_state["identified_domain"] = domain
            st.session_state["show_identify"] = True
            st.session_state["show_advice"] = False

    if click_advice:
        if not combined_text:
            st.warning("请先输入内容或上传文件。")
        else:
            risks = st.session_state.get("identified_risks")
            if not risks:
                st.info("尚未识别风险，请先点击「识别 AI 伦理风险」。")
            else:
                with st.spinner("正在生成风险管理建议…"):
                    result = st.session_state["risk_service"].advise(
                        risks,
                        domain,
                        enterprise_profile=st.session_state.get("enterprise_risk_profile"),
                    )
                st.session_state["advice_result"] = result
                st.session_state["show_advice"] = True
                st.session_state["show_identify"] = False

    if st.session_state.get("show_identify"):
        st.subheader("🔍 风险识别结果")
        # 重点领域：在正式回答上方以深灰小字展示 ERS 说明
        if is_focus:
            st.markdown(
                f'<p style="color:#555555;font-size:0.85rem;line-height:1.6;">{html.escape(ERS_DESCRIPTION)}</p>',
                unsafe_allow_html=True,
            )
        identified = st.session_state["identified_risks"]
        st.markdown(_format_risk_result_for_display(identified))

    if st.session_state.get("show_advice"):
        st.subheader("📋 风险管理建议")
        advice = st.session_state.get("advice_result")
        if advice:
            st.markdown(
                _format_advice_result_for_display(
                    advice,
                    st.session_state["identified_risks"],
                )
            )


def render_enterprise_profile_tab():
    st.subheader("企业 AI 伦理风险画像")
    st.caption("按论文第四章公式计算暴露倍数、企业综合 ERS 和四象限；第五章策略由象限确定性映射。")

    basic_col, patent_col = st.columns(2)
    with basic_col:
        total_patents = st.number_input("企业专利总数", min_value=0, step=1, key="profile_total_patents")
        annual_report_words = st.number_input("企业年报总词数", min_value=0, step=1, key="profile_annual_report_words")
        ai_word_frequency = st.number_input("企业年报 AI 关键词总词频", min_value=0, step=1, key="profile_ai_word_frequency")
    with patent_col:
        personalized_algorithm_patents = st.number_input("个性化算法相关专利数量", min_value=0, step=1, key="profile_personalized_patents")
        machine_vision_patents = st.number_input("机器视觉相关专利数量", min_value=0, step=1, key="profile_vision_patents")
        autonomous_driving_patents = st.number_input("自动驾驶相关专利数量", min_value=0, step=1, key="profile_driving_patents")
        service_robot_patents = st.number_input("服务机器人相关专利数量", min_value=0, step=1, key="profile_robot_patents")

    if st.button("开始企业风险评估", type="primary", key="evaluate_enterprise_risk"):
        try:
            profile = evaluate_enterprise_risk(
                total_patents=total_patents,
                personalized_algorithm_patents=personalized_algorithm_patents,
                machine_vision_patents=machine_vision_patents,
                autonomous_driving_patents=autonomous_driving_patents,
                service_robot_patents=service_robot_patents,
                annual_report_words=annual_report_words,
                ai_word_frequency=ai_word_frequency,
            )
        except EnterpriseRiskValidationError as exc:
            st.error(str(exc))
        else:
            st.session_state["enterprise_risk_profile"] = profile
            st.success("企业风险画像已生成，并将在风险管理建议中作为锁定上下文使用。")

    profile = st.session_state.get("enterprise_risk_profile")
    if not profile:
        st.info("填写基础数据后点击“开始企业风险评估”进行计算。")
        return

    metric_rows = [
        ("专利暴露倍数", profile["patent_exposure"]),
        ("词频暴露倍数", profile["word_exposure"]),
        ("综合暴露倍数", profile["exposure_multiple"]),
        ("AI 专利 ERS", profile["patent_ers"]),
        ("AI 词频 ERS", profile["word_ers"]),
        ("企业综合 ERS", profile["enterprise_ers"]),
    ]
    for row in (metric_rows[:3], metric_rows[3:]):
        columns = st.columns(3)
        for column, (label, value) in zip(columns, row):
            column.metric(label, f"{value:.2f}")
    st.metric("企业风险象限", profile["quadrant_label"])
    st.info(profile["explanation"])
    st.caption(
        f"判定阈值：暴露倍数 90% 分位数 {profile['exposure_threshold']:.2f}；"
        f"企业综合 ERS 均值 {profile['ers_threshold']:.2f}。"
        f"注：判定阈值来自于报告中的146家企业数据。"
    )

    try:
        st.pyplot(create_quadrant_figure(profile), width="stretch")
    except ImportError:
        st.warning("当前环境未安装 matplotlib，暂无法显示四象限图；计算结果不受影响。")

    st.subheader("推荐风险管理策略")
    for strategy in profile["strategies"]:
        st.markdown(
            f"**【{_md_escape(strategy['name'])}】**\n\n{_md_escape(strategy['description'])}"
        )

    quadrant = profile.get("quadrant", "")
    if not quadrant:
        match = re.search(r"Q[1-4]", str(profile.get("quadrant_label", "")))
        quadrant = match.group(0) if match else ""

    if quadrant == "Q1":
        st.warning("贵司整体风险偏高，建议对具体业务进行排查。")


# ---------------------------------------------------------------------------
# 标签页 3：政策栏
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="正在加载政策数据…")
def _cached_policy():
    df = load_policy_df()
    if not df.empty and "发布时间" in df.columns:
        # 尝试转换为 datetime，转换失败则保留原字符串
        df["发布时间_排序"] = pd.to_datetime(df["发布时间"], errors="coerce")
        # 按排序列降序，若全部为 NaT 则按字符串降序
        if df["发布时间_排序"].notna().any():
            df = df.sort_values("发布时间_排序", ascending=False)
        else:
            df = df.sort_values("发布时间", ascending=False)
        df = df.drop(columns=["发布时间_排序"])
    return df


def render_policy_tab():
    df = _cached_policy()
    if df.empty:
        st.warning("暂无政策数据（请确认 data/china_policy.csv 或原始 xls 已就位）。")
        return
    st.caption(f"共 {len(df)} 条政策")

    kw = st.text_input("按标题 / 关键词 / 摘要 / 发布机构筛选", key="policy_filter")
    if st.session_state.get("policy_last_kw") != kw:
        st.session_state["policy_page"] = 1
        st.session_state["policy_last_kw"] = kw
    df = _filter_df(df, kw, ["标题", "关键词", "摘要", "发布机构"])

    if df.empty:
        st.info("没有匹配的政策。")
        return
    start, size = _pagination(len(df), key="policy")
    _render_link_list(df.iloc[start : start + size].to_dict("records"), "原文链接")


# ---------------------------------------------------------------------------
# 标签页 3：资讯栏
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="正在加载资讯数据…")
def _cached_news():
    df = load_news_df()
    if not df.empty and "发布时间" in df.columns:
        df["发布时间_排序"] = pd.to_datetime(df["发布时间"], errors="coerce")
        if df["发布时间_排序"].notna().any():
            df = df.sort_values("发布时间_排序", ascending=False)
        else:
            df = df.sort_values("发布时间", ascending=False)
        df = df.drop(columns=["发布时间_排序"])
    return df


def render_news_tab():
    df = _cached_news()
    if df.empty:
        st.warning("暂无资讯数据（请确认 data/china_news.csv 或原始 xls 已就位）。")
        return
    st.caption(f"共 {len(df)} 条资讯")

    kw = st.text_input("按标题 / 关键词 / 摘要 / 来源筛选", key="news_filter")
    if st.session_state.get("news_last_kw") != kw:
        st.session_state["news_page"] = 1
        st.session_state["news_last_kw"] = kw
    df = _filter_df(df, kw, ["标题", "关键词", "摘要", "来源"])

    if df.empty:
        st.info("没有匹配的资讯。")
        return
    start, size = _pagination(len(df), key="news")
    _render_link_list(df.iloc[start : start + size].to_dict("records"), "原文地址")


# ---------------------------------------------------------------------------
# 页面头部：顶部图片 → 时代之问 → 主标题
# ---------------------------------------------------------------------------
def render_header():
    # 顶部横幅图片
    if BG_IMAGE.exists():
        st.image(str(BG_IMAGE), width="stretch")
    else:
        st.warning("未找到顶部图片 background.jpg（请将其置于项目根目录）。")

    # 时代之问（标题区）
    # st.markdown(
    #     f"""
    #     <div style="text-align:center;padding:0.5rem 0 1.5rem 0;">
    #         <p style="font-size:1.35rem;line-height:1.9;color:#1a1a1a;font-weight:500;margin:0;">
    #             {html.escape(QUOTE).replace(chr(10), '<br/>')}
    #         </p>
    #         <p style="font-size:0.95rem;color:#666;margin-top:0.8rem;">
    #             {html.escape(QUOTE_ATTRIBUTION)}
    #         </p>
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # 主标题
    st.title("AI伦理风险智能评估平台")
    st.caption("输入企业条款、专利或项目内容识别 AI 伦理风险并给出管理建议，同时提供相关政策与资讯的浏览。")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
render_header()

tab_risk, tab_profile, tab_policy, tab_news = st.tabs(
    ["🛡️ 风险识别与建议", "📊 企业风险画像", "📜 政策栏", "📰 资讯栏"]
)

with tab_risk:
    render_risk_tab()

with tab_profile:
    render_enterprise_profile_tab()

with tab_policy:
    render_policy_tab()

with tab_news:
    render_news_tab()
