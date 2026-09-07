import csv
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

XLS_PATH = DATA_DIR / "课题1-001 人工智能伦理治理数据库数据集.xls"
XLSX_PATH = DATA_DIR / "近5年AI与社会伦理综合资讯库（全面版）.xlsx"
XLSX_POLICY_PATH = DATA_DIR / "政策库.xlsx"
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

def _extract_xlsx_news_rows() -> list[dict]:
    """从新 xlsx 的“正式资讯库”中提取 region=中国 的行，并映射为 news 列"""
    if not XLSX_PATH.exists():
        return []
    df = pd.read_excel(XLSX_PATH, sheet_name="正式资讯库", dtype=str).fillna("")
    # 筛选中国地区
    df = df[df["region"] == "中国"]
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "标题": row.get("title", ""),
            "来源": row.get("source", ""),
            "发布时间": row.get("published_date", ""),
            "资讯类型": row.get("content_type", ""),      # 直接使用 content_type（如"风险事件"）
            "关键词": row.get("keywords", ""),
            "摘要": row.get("summary", ""),
            "原文地址": row.get("url", ""),
        })
    return rows

def _merge_news(new_rows: list[dict]) -> None:
    """将新行合并到现有 NEWS_CSV，按原文地址去重（保留已有）"""
    existing = []
    existing_urls = set()
    if NEWS_CSV.exists():
        with NEWS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.append(row)
                existing_urls.add(row.get("原文地址", ""))
    # 过滤掉已有链接的新行
    new_rows_filtered = [r for r in new_rows if r["原文地址"] not in existing_urls]
    if not new_rows_filtered:
        return
    all_rows = existing + new_rows_filtered
    fields = ["标题", "来源", "发布时间", "资讯类型", "关键词", "摘要", "原文地址"]
    _write_csv(all_rows, NEWS_CSV, fields)

def _extract_xlsx_policy_rows() -> list[dict]:
    """读取政策库.xlsx的Sheet1，映射为china_policy.csv的字段"""
    if not XLSX_POLICY_PATH.exists():
        return []
    df = pd.read_excel(XLSX_POLICY_PATH, sheet_name="Sheet1", dtype=str).fillna("")
    rows = []
    for _, row in df.iterrows():
        # 若文件名称或官方链接为空则跳过
        title = row.get("文件/规范名称", "").strip()
        link = row.get("官方链接", "").strip()
        if not title or not link:
            continue
        rows.append({
            "标题": title,
            "发布机构": row.get("发布主体", ""),
            "发布时间": row.get("时间", ""),          # 可能是"2026"或"2026-05"等格式
            "政策类别": row.get("类型", ""),
            "关键词": "",                            # xlsx 无此列，留空
            "摘要": row.get("核心内容及与AI伦理风险治理关联", ""),
            "原文链接": link,
        })
    return rows

def _merge_policy(new_rows: list[dict]) -> None:
    """合并新行到现有 POLICY_CSV，按原文链接去重"""
    existing = []
    existing_links = set()
    if POLICY_CSV.exists():
        with POLICY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.append(row)
                existing_links.add(row.get("原文链接", ""))
    # 过滤掉已有链接的行
    filtered = [r for r in new_rows if r["原文链接"] not in existing_links]
    if not filtered:
        return
    all_rows = existing + filtered
    fields = ["标题", "发布机构", "发布时间", "政策类别", "关键词", "摘要", "原文链接"]
    _write_csv(all_rows, POLICY_CSV, fields)

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

    xlsx_rows = _extract_xlsx_news_rows()
    if xlsx_rows:
        _merge_news(xlsx_rows)

    xlsx_rows1 = _extract_xlsx_policy_rows()
    if xlsx_rows1:
        _merge_policy(xlsx_rows1)


def _drop_empty_link(df: pd.DataFrame, link_col: str) -> pd.DataFrame:
    """过滤掉原文链接为空的行（无链接则无法跳转）。"""
    if df.empty:
        return df
    return df[df[link_col].astype(str).str.strip() != ""]


def _sort_by_time_desc(df: pd.DataFrame, time_col: str = "发布时间") -> pd.DataFrame:
    """按时间列降序排序（时间越近越靠前）。

    时间列可能包含 YYYY / YYYY-MM / YYYY-MM-DD 等格式，或少量非标准值。
    用正则提取年、月、日做排序键，无法解析的值排在最后。
    """
    if df.empty or time_col not in df.columns:
        return df

    def _key(v: str) -> tuple:
        s = str(v).strip()
        m = re.search(r"(\d{4})[-/.年]?(\d{1,2})?[-/.月]?(\d{1,2})?", s)
        if not m:
            return (0, 0, 0)
        year = int(m.group(1))
        month = int(m.group(2)) if m.group(2) else 0
        day = int(m.group(3)) if m.group(3) else 0
        return (year, month, day)

    # 生成排序键，保留原索引以便回到原顺序
    keys = df[time_col].map(_key)
    order = sorted(range(len(df)), key=lambda i: keys.iloc[i], reverse=True)
    return df.iloc[order].reset_index(drop=True)


def load_policy_df() -> pd.DataFrame:
    """加载「地区 = 中国」的政策数据（按发布时间降序）。"""
    _ensure_cache()
    if not POLICY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(POLICY_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    df = _drop_empty_link(df, "原文链接")
    return _sort_by_time_desc(df, "发布时间")


def load_news_df() -> pd.DataFrame:
    """加载「地区 = 中国」的资讯数据（按发布时间降序）。"""
    _ensure_cache()
    if not NEWS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(NEWS_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    df = _drop_empty_link(df, "原文地址")
    return _sort_by_time_desc(df, "发布时间")


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
