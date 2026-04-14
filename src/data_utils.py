"""数据读取、列名校验与基础预处理。"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import YEAR_COL, PRICE_COL, FEATURE_COLS


REQUIRED_COLS = [YEAR_COL, PRICE_COL, *FEATURE_COLS]


def load_data(data_path: str | Path) -> pd.DataFrame:
    """读取 Excel 数据，并按年份排序后返回。

    Parameters
    ----------
    data_path : str | Path
        Excel 文件路径。

    Returns
    -------
    pd.DataFrame
        经过列校验和排序后的数据框。
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件：{data_path}")

    df = pd.read_excel(data_path)
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"数据缺少以下列：{missing}")

    df = df[REQUIRED_COLS].copy().sort_values(YEAR_COL).reset_index(drop=True)
    return df


def split_xy(df: pd.DataFrame):
    """返回年份列、特征矩阵 X 和目标变量 y。"""
    years = df[YEAR_COL].copy()
    X = df[FEATURE_COLS].copy()
    y = df[PRICE_COL].copy()
    return years, X, y
