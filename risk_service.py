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

# ERS 仅覆盖的四个重点领域
ERS_DOMAINS = ["个性化算法", "机器视觉", "自动驾驶", "服务机器人"]


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
    if domain == GENERIC_DOMAIN or domain not in ERS_DOMAINS:
        return ""
    return build_ers_table(domain)


def _is_ers_domain(domain: str) -> bool:
    """判断是否为 ERS 覆盖的四个重点领域。"""
    return domain in ERS_DOMAINS


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
        is_focus = _is_ers_domain(domain)

        # 通用领域：不引入 ERS 表，检索时也排除 ERS 优先级表文档
        exclude = None if is_focus else [ERS_DOC_SOURCE]

        # 系统级硬规则（必须加入 prompt）
        hard_rules = """
【系统级硬规则】
1. ERS 表示特定 AI 领域中不同伦理风险的相对关注优先级，不代表当前企业一定存在该风险。
2. 风险识别必须优先依据用户输入的企业、项目和技术场景信息。
3. 对每项风险应判断其属于：
   - 已识别风险：当前输入存在直接证据；
   - 潜在风险/待核验风险：ERS 或一般风险知识提示值得关注，但当前信息不足；
   - 当前未发现明显风险：当前信息未显示该风险，或已经存在较充分的控制措施。
4. 不要求输出该领域 ERS 中的全部风险。没有证据支持的风险可以不输出，或列为待核验事项。
5. 不得将用户未提供的企业事实自行补充为既有事实。合理推测必须使用"若……则可能……""建议进一步核验……"等条件性表达。
6. 不得因为某项风险 ERS 较高，就直接判断企业实际风险较高。
7. 不得自行生成没有参考依据的具体数字，包括时限、百分比、阈值、赔偿比例、费率、保费折扣、事故指标等。
8. 不得自行创造报告中不存在的新模型、新指数、新保险产品或定价规则。
9. ERS 仅适用于个性化算法、机器视觉、自动驾驶和服务机器人四个重点领域。
10. 当用户选择"其他通用领域"时，不得调用 ERS 分数、排名、Top 5 覆盖率或稳健性分层，只进行一般性 AI 伦理风险识别。
11. 风险管理建议应优先针对已经得到场景证据支持的风险，不得为了覆盖全部 ERS 风险而强行提出治理措施。
12. 用户已经采取的风险控制措施必须被识别并纳入判断，不得忽略已有控制后继续按"完全未治理"状态生成建议。
13. 风险管理建议以必要、直接、可执行为原则。每项已识别风险原则上给出 2-3 项核心建议即可，避免堆砌通用 AI 治理措施。
14. 保险仅作为剩余风险转移工具。不得默认所有伦理风险都可以保险化，也不得自行承诺监管罚款、罚金等一定属于保险责任。
"""

        if is_focus:
            system = (
                "你是一名 AI 伦理风险评估专家。请基于以下参考资料，"
                "识别用户提供的企业条款、专利或项目中可能存在的 AI 伦理风险，"
                "并优先结合给定的重点领域与 ERS 优先级表进行分析。\n\n"
                "【当前重点领域】{domain}\n\n"
                "【AI 伦理风险优先级表（ERS）】\n{ers_table}\n\n"
                "【从知识库检索到的参考资料】\n{context}\n\n"
                + hard_rules
            )
            human = (
                "请识别以下内容中存在的 AI 伦理风险（重点领域：{domain}）：\n{input}\n\n"
                "输出要求：\n"
                "1. 逐条列出识别出的风险，每条包含：风险类型、具体表现、成因；\n"
                "2. 对每条风险必须明确标注其风险等级：\n"
                "   - 【已识别风险】：输入信息能够直接支持；\n"
                "   - 【潜在风险/待核验风险】：ERS 提示值得关注，但现有信息不足以确认；\n"
                "   - 【当前未发现明显风险】：输入中已经存在较充分的控制措施，或没有发现相关迹象。\n"
                "3. 对于【已识别风险】，务必标注其在 ERS 优先级表中对应的 ERS 分数"
                "（即该领域下对应风险维度的「ERS指数化」数值，如「ERS 分数：84.95」），"
                "分数须严格取自上方 ERS 优先级表，不得自行编造；\n"
                "4. 对于【潜在风险/待核验风险】，不得标注 ERS 分数，仅作为风险提示；\n"
                "5. 对于【当前未发现明显风险】，简要说明已有控制措施即可；\n"
                "6. 优先按风险等级排序（已识别 > 待核验 > 未发现），同等级内按 ERS 分数从高到低排序；\n"
                "7. 若某项风险不在该领域的 ERS 表中，可结合参考资料补充，但不要标注分数。\n\n"
                "注意：不要输出保险、保费、承保、再保险等风险转移或保险协同内容。\n"
                "注意：不得将用户未提供的信息写成企业当前事实。对于合理但未经输入支持的风险，"
                "只能使用条件性表述。\n"
                "注意：不要求输出该领域 ERS 中的全部五类风险，没有证据支持的风险可以不输出。"
            )
        else:
            system = (
                "你是一名 AI 伦理风险评估专家。请基于以下参考资料，"
                "识别用户提供的企业条款、专利或项目中可能存在的 AI 伦理风险。\n\n"
                "【从知识库检索到的参考资料】\n{context}\n\n"
                + hard_rules
            )
            human = (
                "请识别以下内容中存在的 AI 伦理风险：\n{input}\n\n"
                "注意：直接给出潜在的 AI 伦理风险即可，不要提及 ERS、优先级指数等概念，"
                "也不要输出保险、保费、承保、再保险等风险转移或保险协同内容。\n"
                "注意：不得将用户未提供的信息写成企业当前事实。对于合理但未经输入支持的风险，"
                "只能使用条件性表述。\n"
                "注意：不得自行生成没有参考依据的具体数字，包括时限、百分比、阈值等。\n"
                "注意：当前场景不属于 ERS 模型已覆盖的四个重点领域，因此以下内容仅进行一般性"
                "AI 伦理风险识别，不提供 ERS 分数和风险排序。"
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
        is_focus = _is_ers_domain(domain)

        # 系统级硬规则（必须加入 prompt）
        hard_rules = """
【系统级硬规则】
1. 不得自行生成没有参考依据的具体数字，包括时限、百分比、阈值、赔偿比例、费率、保费折扣、事故指标等。
2. 不得自行创造报告中不存在的新模型、新指数、新保险产品或定价规则。
3. 风险管理建议应优先针对已经得到场景证据支持的风险，不得为了覆盖全部 ERS 风险而强行提出治理措施。
4. 用户已经采取的风险控制措施必须被识别并纳入判断，不得忽略已有控制后继续按"完全未治理"状态生成建议。
5. 风险管理建议以必要、直接、可执行为原则。每项已识别风险原则上给出 2-3 项核心建议即可，避免堆砌通用 AI 治理措施。
6. 保险仅作为剩余风险转移工具。不得默认所有伦理风险都可以保险化，也不得自行承诺监管罚款、罚金等一定属于保险责任。
7. 不得将用户未提供的企业事实自行补充为既有事实。合理推测必须使用条件性表述。
8. 对"已识别风险"给出 2-3 项最重要、最直接的控制措施。
9. 对"待核验风险"不要直接提出一整套治理体系，而是告诉用户需要核验什么，例如："核验不同群体误识率是否存在显著差异。"
10. 对"当前控制较好的风险"明确肯定现有控制："当前已有本地处理和短期删除机制，建议持续监测，不需要新增复杂措施。"
11. 保险建议不要每个风险都自动推荐保险，只在最后增加一个统一的"风险转移建议"，说明剩余风险在满足可识别、可验证、可计量等条件后，可以进一步考虑保险转移。
12. 当用户选择"其他通用领域"时，不得调用 ERS 分数、排名、Top 5 覆盖率或稳健性分层。
"""

        system = (
            "你是一名 AI 伦理风险管理顾问。请基于以下参考资料，"
            "针对已识别出的 AI 伦理风险，给出具体、可落地的风险管理建议。\n\n"
        )
        if is_focus:
            system += (
                "【当前重点领域】{domain}\n\n"
                "【AI 伦理风险优先级表（ERS）】\n{ers_table}\n\n"
            )
        system += (
            "【从知识库检索到的参考资料】\n{context}\n\n"
            + hard_rules
        )

        if is_focus:
            human = (
                "企业相关内容：\n{input}\n\n"
                "已识别出的 AI 伦理风险：\n{risks}\n\n"
                "请给出风险管理建议（可结合保险等风险转移与协同机制）。\n\n"
                "注意：不得自行生成没有参考依据的具体数字。\n"
                "注意：对已有控制措施的风险应予以肯定，不要重复建议。\n"
                "注意：保险建议统一在最后给出，不要每个风险都推荐保险。"
            )
        else:
            human = (
                "企业相关内容：\n{input}\n\n"
                "已识别出的 AI 伦理风险：\n{risks}\n\n"
                "请给出风险管理建议。\n\n"
                "注意：当前场景不属于 ERS 模型已覆盖的四个重点领域，"
                "因此不提供 ERS 分数和风险排序。\n"
                "注意：不得自行生成没有参考依据的具体数字。\n"
                "注意：不得自行创造新模型、新指数、新保险产品或定价规则。\n"
                "注意：保险建议统一在最后给出，不要每个风险都推荐保险。"
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