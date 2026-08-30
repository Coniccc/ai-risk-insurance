import html
import os
from pathlib import Path

import streamlit as st

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

from data_loader import (  # noqa: E402
    DOMAINS,
    ERS_DESCRIPTION,
    GENERIC_DOMAIN,
    load_data_if_needed,
)
from data_source import load_news_df, load_policy_df  # noqa: E402
from risk_service import RiskService  # noqa: E402

# 领域选择列表（4 个重点领域 + 其他通用领域）
DOMAIN_OPTIONS = DOMAINS + [GENERIC_DOMAIN]


# ---------------------------------------------------------------------------
# 渲染工具
# ---------------------------------------------------------------------------
def _md_escape(s: str) -> str:
    return html.escape(str(s))


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
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    page = int(st.session_state[page_key])
    page = min(max(1, page), total_pages)

    c1, c2, c3 = st.columns([1, 1, 3])
    if c1.button("⬅ 上一页", disabled=(page <= 1), key=f"{key}_prev", width="stretch"):
        page = page - 1
        st.session_state[page_key] = page
    if c2.button("下一页 ➡", disabled=(page >= total_pages), key=f"{key}_next", width="stretch"):
        page = page + 1
        st.session_state[page_key] = page
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
        st.session_state["identified_risks"] = ""

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

    col_identify, col_advice = st.columns(2)
    with col_identify:
        click_identify = st.button("🔍 识别 AI 伦理风险", type="primary", width="stretch")
    with col_advice:
        click_advice = st.button("📋 生成风险管理建议", width="stretch")

    if click_identify:
        if not user_text.strip():
            st.warning("请先输入需要识别的企业条款、专利或项目内容。")
        else:
            with st.spinner("正在识别 AI 伦理风险…"):
                result = st.session_state["risk_service"].identify_risks(user_text, domain)
            st.session_state["identified_risks"] = result
            st.session_state["identified_domain"] = domain
            st.session_state["show_identify"] = True
            st.session_state["show_advice"] = False

    if click_advice:
        if not user_text.strip():
            st.warning("请先输入企业条款、专利或项目内容。")
        else:
            risks = st.session_state.get("identified_risks", "")
            if not risks:
                st.info("尚未识别风险，请先点击「识别 AI 伦理风险」；或直接基于原文生成建议。")
            with st.spinner("正在生成风险管理建议…"):
                result = st.session_state["risk_service"].advise(user_text, risks, domain)
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
        st.markdown(st.session_state["identified_risks"])

    if st.session_state.get("show_advice"):
        st.subheader("📋 风险管理建议")
        st.markdown(st.session_state.get("advice_result", ""))


# ---------------------------------------------------------------------------
# 标签页 2：政策栏
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="正在加载政策数据…")
def _cached_policy():
    return load_policy_df()


def render_policy_tab():
    df = _cached_policy()
    if df.empty:
        st.warning("暂无政策数据（请确认 data/china_policy.csv 或原始 xls 已就位）。")
        return
    st.caption(f"共 {len(df)} 条「地区 = 中国」的政策")

    kw = st.text_input("按标题 / 关键词 / 摘要 / 发布机构筛选", key="policy_filter")
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
    return load_news_df()


def render_news_tab():
    df = _cached_news()
    if df.empty:
        st.warning("暂无资讯数据（请确认 data/china_news.csv 或原始 xls 已就位）。")
        return
    st.caption(f"共 {len(df)} 条「地区 = 中国」的资讯")

    kw = st.text_input("按标题 / 关键词 / 摘要 / 来源筛选", key="news_filter")
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
    st.markdown(
        f"""
        <div style="text-align:center;padding:0.5rem 0 1.5rem 0;">
            <p style="font-size:1.35rem;line-height:1.9;color:#1a1a1a;font-weight:500;margin:0;">
                {html.escape(QUOTE).replace(chr(10), '<br/>')}
            </p>
            <p style="font-size:0.95rem;color:#666;margin-top:0.8rem;">
                {html.escape(QUOTE_ATTRIBUTION)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 主标题
    st.title("AI 伦理风险识别与管理建议")
    st.caption("输入企业条款、专利或项目内容识别 AI 伦理风险并给出管理建议,同时提供中国地区相关政策与资讯的浏览。")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
render_header()

tab_risk, tab_policy, tab_news = st.tabs(["🛡️ 风险识别与建议", "📜 政策栏", "📰 资讯栏"])

with tab_risk:
    render_risk_tab()

with tab_policy:
    render_policy_tab()

with tab_news:
    render_news_tab()
