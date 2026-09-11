import csv
from pathlib import Path

from knowledge_base import KnowledgeBaseService

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 载入知识库的纯文本资料（ERS v3 旧口径说明已移除，由 1.2.0 正式口径资料取代）
TXT_FILES = [
    "A思路--风险管理方案部分综述.txt",
    "指数保险设计方案.txt",
    "天书.txt",
    "模型方法与修订说明.txt",
    "第三章_ERS风险识别模型_1.2.0最终版.txt",
    "ers描述.txt",
]

# ERS 结果表（1.2.0 正式口径）
ERS_RESULTS_CSV = "ers_results.csv"

ERS_RISK_TYPE_TO_CSV_NAME = {
    "autonomy": "自主决策",
    "privacy": "隐私",
    "bias": "偏见歧视",
    "fairness": "公平",
    "accountability": "责任",
}

# ERS 概念描述文件（用于界面展示）
ERS_DESC_FILE = "ers描述.txt"

# 四个重点 AI 领域（与 ERS 优先级表一致）
DOMAINS = ["个性化算法", "机器视觉", "自动驾驶", "服务机器人"]
GENERIC_DOMAIN = "其他通用领域"


def read_text_auto(path):
    """读取文本文件，自动适配 UTF-8 / GB18030 编码。"""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


# ERS 概念说明（在重点领域识别结果上方以深灰小字展示）。
# 优先读取 data/ers描述.txt 中的权威定义；文件缺失时回退到内置说明。
_FALLBACK_ERS_DESCRIPTION = (
    "ERS（Ethical Risk Score）是融合了多源公开证据、专家与公众判断及证据充分度修正，"
    "对四类 AI 应用领域中的五类核心伦理风险进行综合识别与排序，"
    "形成的 20 个「领域—风险」组合的相对关注优先级。"
)


def _load_ers_description() -> str:
    path = DATA_DIR / ERS_DESC_FILE
    if path.exists():
        text = read_text_auto(path).strip()
        if text:
            return text
    return _FALLBACK_ERS_DESCRIPTION


ERS_DESCRIPTION = _load_ers_description()


def build_ers_table(domain: str | None = None) -> str:
    """把 ers_results.csv 整理成供识别链路参考的优先级表（Markdown 文本）。

    domain 为 None 时返回全部领域；传入某个领域名时仅返回该领域的行。
    """
    csv_path = DATA_DIR / ERS_RESULTS_CSV
    if not csv_path.exists():
        return ""

    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if domain and row.get("领域") != domain:
                continue
            rows.append(row)

    if not rows:
        return ""

    scope = f"领域：{domain}" if domain else "4 个 AI 重点领域"
    lines = [
        f"AI 伦理风险基准优先级表（ERS，{scope}）",
        "ERS 是风险关注优先级指数，数值越大表示该「领域-风险」组合越值得优先关注。",
        "",
        "| 排名 | 领域 | 风险维度 | ERS指数化 |",
        "|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['全局排名']} | {row['领域']} | {row['风险维度']} "
            f"| {float(row['ERS_指数化']):.2f} |"
        )
    return "\n".join(lines)


def get_ers_score(domain: str, risk_type: str) -> str | None:
    """Return the authoritative ERS score for one domain/risk pair.

    This is deliberately a precise CSV lookup rather than a prompt-oriented
    table.  Unsupported domains, unsupported risk types, missing files, and
    missing rows all return ``None``.
    """
    if domain not in DOMAINS:
        return None

    risk_name = ERS_RISK_TYPE_TO_CSV_NAME.get(risk_type)
    if risk_name is None:
        return None

    csv_path = DATA_DIR / ERS_RESULTS_CSV
    if not csv_path.exists():
        return None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("领域") == domain and row.get("风险简称") == risk_name:
                score = row.get("ERS_指数化")
                if score is None:
                    return None
                # Preserve the published value while normalising its display
                # precision to the existing ERS-table convention.
                return f"{float(score):.2f}"
    return None


def build_corpus() -> list[tuple[str, str]]:
    """构造 (名称, 文本) 列表，用于批量载入知识库。"""
    corpus = []

    # 纯文本资料
    for name in TXT_FILES:
        path = DATA_DIR / name
        if path.exists():
            corpus.append((name, read_text_auto(path)))

    # ERS 结果表单独作为一个结构化知识块
    table = build_ers_table()
    if table:
        corpus.append(("ERS优先级表", table))

    return corpus


def load_data_if_needed(force=False):
    """
    把 data 目录资料载入向量库。

    返回 (是否执行了载入, 结果摘要字符串)。
    默认通过 md5 去重，已载入过的内容会跳过；force=True 时强制重载。
    """
    from knowledge_base import get_string_md5, check_md5

    service = KnowledgeBaseService()
    corpus = build_corpus()
    if not corpus:
        return False, "[跳过]data 目录下未找到可载入的资料"

    summary = []
    loaded = 0
    for name, text in corpus:
        if not text.strip():
            continue
        md5_hex = get_string_md5(text)
        if not force and check_md5(md5_hex):
            summary.append(f"[跳过]已存在：{name}")
            continue
        service.upload_by_str(text, name)
        summary.append(f"[成功]已载入：{name}")
        loaded += 1

    summary.insert(0, f"共 {loaded} 条资料载入知识库")
    return loaded > 0, "\n".join(summary)


if __name__ == "__main__":
    changed, msg = load_data_if_needed()
    print(msg)
