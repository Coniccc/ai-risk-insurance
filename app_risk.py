"""
AI 伦理风险识别与管理建议
========================
用户在企业条款 / 专利 / 项目文本输入框内填写内容，
通过两个独立按钮分别触发：
    1. 识别 AI 伦理风险
    2. 生成风险管理建议
数据来源为 data 目录下的资料（见 data_loader.py）。
"""
import streamlit as st

from data_loader import load_data_if_needed
from risk_service import RiskService

st.set_page_config(page_title="AI 伦理风险识别与管理建议", page_icon="🛡️")

st.title("AI 伦理风险识别与管理建议")
st.caption("输入企业的相关条款、专利或项目内容，识别其中存在的 AI 伦理风险并给出风险管理建议。")

# 载入数据资料到向量库（md5 去重，重复打开页面不会重复写入）
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = True

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
