import streamlit as st
from rag import RagService
import config_data as config

# 标题
st.title("智能问答")
st.divider()    # 分隔符

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

if "message" not in st.session_state:
    st.session_state["message"] = [{"role":"assistant","content":"你好，有什么可以帮助你"}]

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 在页面最下方提供用户输入栏
prompt = st.chat_input()
if prompt:
    # 在页面输出用户提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    ai_res_list = []

    with st.spinner("AI思考中..."):
        # res = st.session_state["rag"].chain.invoke({"input":prompt},config.session_config)
        res_stream = st.session_state["rag"].chain.stream({"input":prompt},config.session_config)

        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})
        # st.session_state["message"].append({"role":"assistant","content":res})

        # ["a", "b", "c"]   "".join(list)    -> abc
        # ["a", "b", "c"]   ",".join(list)    -> a,b,c
