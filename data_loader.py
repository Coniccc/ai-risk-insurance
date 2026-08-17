"""
数据加载服务
============
负责把 data 文件夹里的资料解析、编码统一后，批量载入向量数据库，
供 AI 伦理风险识别 / 风险管理建议两条链路检索使用。

data 文件夹内文件编码不统一：
    - *.csv / *.py / ERS_*计算说明.txt 为 UTF-8
    - A思路--*.txt / 指数保险设计方案.txt 为 GBK(GB18030)
因此统一按「utf-8 优先、失败回退 gb18030」读取。
"""
import csv
import os
from pathlib import Path

from knowledge_base import KnowledgeBaseService

# 项目根目录（本文件所在目录）
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 优先载入的纯文本资料（文件名 -> 是否作为独立知识块）
TXT_FILES = [
    "A思路--风险管理方案部分综述.txt",
    "指数保险设计方案.txt",
    "ERS_v3_4x5_计算说明.txt",
]


def read_text_auto(path):
    """读取文本文件，自动适配 UTF-8 / GB18030 编码。"""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def build_ers_table() -> str:
    """把 ers_v3_results.csv 整理成供识别链路参考的优先级表（Markdown 文本）。"""
    csv_path = DATA_DIR / "ers_v3_results.csv"
    if not csv_path.exists():
        return ""

    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    lines = [
        "AI 伦理风险基准优先级表（ERS，来源 4 个 AI 领域 × 5 类伦理风险）",
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
