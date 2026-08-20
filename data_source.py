import csv
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

XLS_PATH = DATA_DIR / "课题1-001 人工智能伦理治理数据库数据集.xls"
POLICY_CSV = DATA_DIR / "china_policy.csv"
NEWS_CSV = DATA_DIR / "china_news.csv"

# 政策 sheet 的列映射：目标列名 -> (xls 列索引, 空值默认)
POLICY_COL_MAP = {
    "标题": (0, ""),
    "发布机构": (3, ""),
    "发布时间": (4, ""),
    "政策类别": (5, ""),
    "关键词": (7, ""),
    "摘要": (10, ""),
    "原文链接": (9, ""),
}

# 资讯 sheet 的列映射
NEWS_COL_MAP = {
    "标题": (0, ""),
    "来源": (9, ""),
    "发布时间": (6, ""),
    "资讯类型": (11, ""),
    "关键词": (8, ""),
    "摘要": (10, ""),
    "原文地址": (13, ""),
}

# 资讯类型：1 新闻，2 案例（xls 中存储为 "1.0" / "2.0"）
NEWS_TYPE_MAP = {"1": "新闻", "2": "案例", "1.0": "新闻", "2.0": "案例"}


def _cell(v) -> str:
    """把 xlrd 单元格值统一转为去除首尾空白的字符串。"""
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s == "None" else s


def _extract_china_rows(sheet_name: str, col_map: dict) -> list[dict]:
    """从指定 sheet 中筛出「地区 = 中国」的行。"""
    import xlrd

    book = xlrd.open_workbook(str(XLS_PATH), on_demand=True)
    sh = book.sheet_by_name(sheet_name)
    rows = []
    for r in range(1, sh.nrows):
        region = _cell(sh.cell_value(r, 2))
        if region != "中国":
            continue
        row = {field: _cell(sh.cell_value(r, idx)) for field, (idx, _) in col_map.items()}
        # 资讯类型转中文
        if "资讯类型" in row:
            row["资讯类型"] = NEWS_TYPE_MAP.get(row["资讯类型"], row["资讯类型"])
        rows.append(row)
    book.release_resources()
    return rows


def _write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_cache() -> None:
    """若轻量 CSV 缺失且存在原始 xls，则现场提取生成缓存。"""
    if POLICY_CSV.exists() and NEWS_CSV.exists():
        return
    if not XLS_PATH.exists():
        return

    policy_rows = _extract_china_rows("政策", POLICY_COL_MAP)
    news_rows = _extract_china_rows("资讯", NEWS_COL_MAP)
    _write_csv(policy_rows, POLICY_CSV, list(POLICY_COL_MAP.keys()))
    _write_csv(news_rows, NEWS_CSV, list(NEWS_COL_MAP.keys()))


def _drop_empty_link(df: pd.DataFrame, link_col: str) -> pd.DataFrame:
    """过滤掉原文链接为空的行（无链接则无法跳转）。"""
    if df.empty:
        return df
    return df[df[link_col].astype(str).str.strip() != ""]


def load_policy_df() -> pd.DataFrame:
    """加载「地区 = 中国」的政策数据。"""
    _ensure_cache()
    if not POLICY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(POLICY_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    return _drop_empty_link(df, "原文链接")


def load_news_df() -> pd.DataFrame:
    """加载「地区 = 中国」的资讯数据。"""
    _ensure_cache()
    if not NEWS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(NEWS_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    return _drop_empty_link(df, "原文地址")


def search_policy_news(keyword: str) -> pd.DataFrame:
    """在政策与资讯中按关键词搜索（匹配标题 / 关键词 / 摘要 / 来源或机构）。"""
    kw = (keyword or "").strip().lower()
    if not kw:
        return pd.DataFrame()

    policy = load_policy_df()
    news = load_news_df()

    hits = []

    def _match(row, extra_fields):
        text = " ".join([str(row.get(f, "")) for f in extra_fields]).lower()
        return kw in text

    if not policy.empty:
        for _, row in policy.iterrows():
            if _match(row, ["标题", "关键词", "摘要", "发布机构"]):
                hits.append(
                    {
                        "类型": "政策",
                        "标题": row["标题"],
                        "发布时间": row["发布时间"],
                        "来源": row["发布机构"],
                        "链接": row["原文链接"],
                    }
                )

    if not news.empty:
        for _, row in news.iterrows():
            if _match(row, ["标题", "关键词", "摘要", "来源"]):
                hits.append(
                    {
                        "类型": "资讯",
                        "标题": row["标题"],
                        "发布时间": row["发布时间"],
                        "来源": row["来源"],
                        "链接": row["原文地址"],
                    }
                )

    return pd.DataFrame(hits, columns=["类型", "标题", "发布时间", "来源", "链接"])


if __name__ == "__main__":
    _ensure_cache()
    print("政策条数:", len(load_policy_df()))
    print("资讯条数:", len(load_news_df()))
