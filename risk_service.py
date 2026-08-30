from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

import config_data as config
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from vector_stores import VectorStoreService
from data_loader import GENERIC_DOMAIN, build_ers_table

# 识别链路检索更多文档、建议链路也放宽，保证上下文充分
RISK_RETRIEVE_K = 6
ADVICE_RETRIEVE_K = 6


def _format_docs(docs: list[Document], exclude_sources: list[str] | None = None) -> str:
    """把检索到的文档片段拼接成字符串，可按来源排除指定文档。"""
    if not docs:
        return "无相关资料"
    exclude = set(exclude_sources or [])
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "未知来源")
        if src in exclude:
            continue
        parts.append(f"[{src}]\n{doc.page_content}")
    if not parts:
        return "无相关资料"
    return "\n\n---\n\n".join(parts)


# 通用领域需要排除 ERS 表文档，避免结果中出现 ERS
ERS_DOC_SOURCE = "ERS优先级表"


def _pick_input(x: dict) -> str:
    """从链的输入 dict 中取文本。"""
    return x["input"]


def _pick_domain(x: dict) -> str:
    """从链的输入 dict 中取领域。"""
    return x.get("domain", GENERIC_DOMAIN)


def _ers_for_domain(x: dict) -> str:
    """重点领域返回该领域的 ERS 表，通用领域返回空串。"""
    domain = x.get("domain", GENERIC_DOMAIN)
    if domain == GENERIC_DOMAIN:
        return ""
    return build_ers_table(domain)


class RiskService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.chat_model = ChatTongyi(model=config.chat_model_name)

    # ------------------------------------------------------------------
    # 风险识别链
    # ------------------------------------------------------------------
    def __get_identify_chain(self, domain: str):
        """构造识别链。重点领域注入 ERS 表；通用领域不注入且排除 ERS 文档。"""
        retriever = self.vector_service.get_retriever(k=RISK_RETRIEVE_K)
        is_focus = domain != GENERIC_DOMAIN

        # 通用领域：不引入 ERS 表，检索时也排除 ERS 优先级表文档
        exclude = None if is_focus else [ERS_DOC_SOURCE]

        if is_focus:
            system = (
                "你是一名 AI 伦理风险评估专家。请基于以下参考资料，"
                "识别用户提供的企业条款、专利或项目中可能存在的 AI 伦理风险，"
                "并优先结合给定的重点领域与 ERS 优先级表进行分析。\n\n"
                "【当前重点领域】{domain}\n\n"
                "【AI 伦理风险优先级表（ERS）】\n{ers_table}\n\n"
                "【从知识库检索到的参考资料】\n{context}"
            )
            human = (
                "请识别以下内容中存在的 AI 伦理风险（重点领域：{domain}）：\n{input}\n\n"
                "注意：请只输出风险识别结果本身（风险类型、具体表现、成因），"
                "不要输出保险、保费、承保、再保险等风险转移或保险协同内容。"
            )
        else:
            system = (
                "你是一名 AI 伦理风险评估专家。请基于以下参考资料，"
                "识别用户提供的企业条款、专利或项目中可能存在的 AI 伦理风险。\n\n"
                "【从知识库检索到的参考资料】\n{context}"
            )
            human = (
                "请识别以下内容中存在的 AI 伦理风险：\n{input}\n\n"
                "注意：直接给出潜在的 AI 伦理风险即可，不要提及 ERS、优先级指数等概念，"
                "也不要输出保险、保费、承保、再保险等风险转移或保险协同内容。"
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )

        return (
            {
                "input": RunnableLambda(_pick_input),
                "context": RunnableLambda(_pick_input) | retriever | RunnableLambda(lambda docs: _format_docs(docs, exclude)),
                "domain": RunnableLambda(_pick_domain),
                "ers_table": RunnableLambda(_ers_for_domain),
            }
            | prompt
            | self.chat_model
            | StrOutputParser()
        )

    # ------------------------------------------------------------------
    # 管理建议链
    # ------------------------------------------------------------------
    def __get_advice_chain(self, domain: str):
        retriever = self.vector_service.get_retriever(k=ADVICE_RETRIEVE_K)
        is_focus = domain != GENERIC_DOMAIN

        system = (
            "你是一名 AI 伦理风险管理顾问。请基于以下参考资料，"
            "针对已识别出的 AI 伦理风险，给出具体、可落地的风险管理建议。\n\n"
        )
        if is_focus:
            system += (
                "【当前重点领域】{domain}\n\n"
                "【AI 伦理风险优先级表（ERS）】\n{ers_table}\n\n"
            )
        system += "【从知识库检索到的参考资料】\n{context}"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "企业相关内容：\n{input}\n\n"
                    "已识别出的 AI 伦理风险：\n{risks}\n\n"
                    "请给出风险管理建议（可结合保险等风险转移与协同机制）。",
                ),
            ]
        )

        return (
            {
                "input": RunnableLambda(_pick_input),
                "risks": RunnableLambda(lambda x: x.get("risks", "")),
                "context": RunnableLambda(_pick_input) | retriever | _format_docs,
                "domain": RunnableLambda(_pick_domain),
                "ers_table": RunnableLambda(_ers_for_domain),
            }
            | prompt
            | self.chat_model
            | StrOutputParser()
        )

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def identify_risks(self, text: str, domain: str = GENERIC_DOMAIN) -> str:
        """识别文本中的 AI 伦理风险（可按重点领域）。"""
        chain = self.__get_identify_chain(domain)
        return chain.invoke({"input": text, "domain": domain})

    def advise(self, text: str, risks: str, domain: str = GENERIC_DOMAIN) -> str:
        """基于原文与已识别风险，给出管理建议（可含保险协同机制）。"""
        chain = self.__get_advice_chain(domain)
        return chain.invoke({"input": text, "risks": risks, "domain": domain})
