"""
AI 伦理风险识别与管理建议
========================
用户在企业条款 / 专利 / 项目文本输入框内填写内容，
通过两个独立按钮分别触发：
    1. 识别 AI 伦理风险
    2. 生成风险管理建议
数据来源为 data 目录下的资料（见 data_loader.py）。

部署说明（Streamlit Cloud）：
    - 依赖见 requirements.txt
    - DashScope API Key 通过 App 的 Settings → Secrets 配置：
        DASHSCOPE_API_KEY = "sk-xxxx"
"""
import os

import streamlit as st


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

from data_loader import load_data_if_needed  # noqa: E402
from risk_service import RiskService  # noqa: E402

st.set_page_config(page_title="AI 伦理风险识别与管理建议", page_icon="🛡️")

st.title("AI 伦理风险识别与管理建议")
st.caption("输入企业的相关条款、专利或项目内容，识别其中存在的 AI 伦理风险并给出风险管理建议。")


@st.cache_resource(show_spinner="正在载入知识库…")
def _init_knowledge_base():
    """把 data 目录资料嵌入向量库。云端每次启动自动执行一次（运行盘为临时盘）。"""
    if not API_KEY:
        return False, "未配置 DASHSCOPE_API_KEY"
    try:
        _, msg = load_data_if_needed()
        return True, msg
    except Exception as exc:  # noqa: BLE001
        return False, f"知识库载入失败：{exc}"


ok, load_msg = _init_knowledge_base()
if not ok:
    st.error(
        "⚠️ " + load_msg + "\n\n"
        "部署到 Streamlit Cloud 时，请在 App 的 **Settings → Secrets** 中添加：\n\n"
        "```toml\nDASHSCOPE_API_KEY = \"sk-xxxx\"\n```\n\n"
        "本地运行则设置环境变量 `DASHSCOPE_API_KEY` 即可。"
    )
    st.stop()


if "risk_service" not in st.session_state:
    st.session_state["risk_service"] = RiskService()

# 初始化风险识别结果，供「建议」按钮复用
if "identified_risks" not in st.session_state:
    st.session_state["identified_risks"] = ""

st.divider()

user_text = st.text_area(
    "请输入企业条款 / 专利 / 项目内容",
    height=200,
    placeholder="例如：本公司将使用人工智能算法对海量用户数据进行自动化画像与评分……",
)

col_identify, col_advice = st.columns(2)

with col_identify:
    click_identify = st.button("🔍 识别 AI 伦理风险", type="primary", use_container_width=True)

with col_advice:
    click_advice = st.button("📋 生成风险管理建议", use_container_width=True)

if click_identify:
    if not user_text.strip():
        st.warning("请先输入需要识别的企业条款、专利或项目内容。")
    else:
        with st.spinner("正在识别 AI 伦理风险…"):
            result = st.session_state["risk_service"].identify_risks(user_text)
        st.session_state["identified_risks"] = result
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
            result = st.session_state["risk_service"].advise(user_text, risks)
        st.session_state["advice_result"] = result
        st.session_state["show_advice"] = True
        st.session_state["show_identify"] = False

st.divider()

if st.session_state.get("show_identify"):
    st.subheader("🔍 风险识别结果")
    st.markdown(st.session_state["identified_risks"])

if st.session_state.get("show_advice"):
    st.subheader("📋 风险管理建议")
    st.markdown(st.session_state.get("advice_result", ""))
