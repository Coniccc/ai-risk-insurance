"""
AI 伦理风险服务
==============
提供两条独立链路：
    1. 风险识别链   —— 识别用户输入（条款 / 专利 / 项目）中存在的 AI 伦理风险
    2. 管理建议链   —— 针对已识别风险，给出可落地的风险管理建议

两条链路都基于 RAG：从向量库检索 data 目录载入的资料作为上下文，
配合 ERS 优先级表与风险管理工具知识，交给大模型生成结果。
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import config_data as config
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from vector_stores import VectorStoreService
from data_loader import build_ers_table

# 识别链路检索更多文档、建议链路也放宽，保证上下文充分
RISK_RETRIEVE_K = 6
ADVICE_RETRIEVE_K = 6


def _format_docs(docs: list[Document]) -> str:
    """把检索到的文档片段拼接成字符串。"""
    if not docs:
        return "无相关资料"
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "未知来源")
        parts.append(f"[{src}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class RiskService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        # 识别与建议共用同一张 ERS 优先级表
        self.ers_table = build_ers_table()

    # ------------------------------------------------------------------
    # 风险识别链
    # ------------------------------------------------------------------
    def __get_identify_chain(self):
        retriever = self.vector_service.get_retriever(k=RISK_RETRIEVE_K)

        system = (
            "你是一名 AI 伦理风险评估专家。请基于以下参考资料，"
            "识别用户提供的企业条款、专利或项目中可能存在的 AI 伦理风险。\n\n"
            "【AI 伦理风险优先级表】\n{ers_table}\n\n"
            "【从知识库检索到的参考资料】\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "请识别以下内容中存在的 AI 伦理风险：\n{input}"),
            ]
        )

        def _format_for_prompt(value: dict) -> dict:
            return {
                "input": value["input"],
                "context": value["context"],
                "ers_table": value["ers_table"],
            }

        return (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(lambda x: x["input"]) | retriever | _format_docs,
                "ers_table": RunnableLambda(lambda x: self.ers_table or "无"),
            }
            | RunnableLambda(_format_for_prompt)
            | prompt
            | self.chat_model
            | StrOutputParser()
        )

    # ------------------------------------------------------------------
    # 管理建议链
    # ------------------------------------------------------------------
    def __get_advice_chain(self):
        retriever = self.vector_service.get_retriever(k=ADVICE_RETRIEVE_K)

        system = (
            "你是一名 AI 伦理风险管理顾问。请基于以下参考资料，"
            "针对已识别出的 AI 伦理风险，给出具体、可落地的风险管理建议。\n\n"
            "【AI 伦理风险优先级表】\n{ers_table}\n\n"
            "【从知识库检索到的参考资料】\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "企业相关内容：\n{input}\n\n"
                    "已识别出的 AI 伦理风险：\n{risks}\n\n"
                    "请给出风险管理建议。",
                ),
            ]
        )

        def _format_for_prompt(value: dict) -> dict:
            return {
                "input": value["input"],
                "risks": value["risks"],
                "context": value["context"],
                "ers_table": value["ers_table"],
            }

        return (
            {
                "input": RunnablePassthrough(),
                "risks": RunnablePassthrough(),
                "context": RunnableLambda(lambda x: x["input"]) | retriever | _format_docs,
                "ers_table": RunnableLambda(lambda x: self.ers_table or "无"),
            }
            | RunnableLambda(_format_for_prompt)
            | prompt
            | self.chat_model
            | StrOutputParser()
        )

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def identify_risks(self, text: str) -> str:
        """识别文本中的 AI 伦理风险。"""
        chain = self.__get_identify_chain()
        return chain.invoke({"input": text})

    def advise(self, text: str, risks: str) -> str:
        """基于原文与已识别风险，给出管理建议。"""
        chain = self.__get_advice_chain()
        return chain.invoke({"input": text, "risks": risks})
