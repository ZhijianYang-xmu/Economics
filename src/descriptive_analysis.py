"""描述性统计、多重共线性诊断和第1章/第2章图表生成。"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

from .config import (
    YEAR_COL, PRICE_COL, FEATURE_COLS, ABBREVIATIONS,
    FIGURE_TITLES, VARIABLE_SYSTEM, VARIABLE_ABBREVIATIONS,
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.bbox"] = "tight"


def _set_integer_year_ticks(ax, years):
    years = list(years)
    tick_years = years[::2] if len(years) > 10 else years
    if years[-1] not in tick_years:
        tick_years.append(years[-1])
    ax.set_xticks(tick_years)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _base_2001_index(series: pd.Series) -> pd.Series:
    """计算以样本首年为100的指数。"""
    return series / series.iloc[0] * 100


def build_variable_system_table() -> pd.DataFrame:
    """生成 Table 2-1：变量体系与理论预期。"""
    return pd.DataFrame(
        VARIABLE_SYSTEM,
        columns=["因素类别", "变量名称", "英文简称", "理论预期方向", "理论说明"],
    )


def build_descriptive_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """生成 Table 2-2：描述性统计。"""
    numeric_cols = [PRICE_COL, *FEATURE_COLS]
    desc = df[numeric_cols].describe().T[["count", "mean", "std", "min", "max"]]
    desc = desc.rename(columns={
        "count": "样本数",
        "mean": "均值",
        "std": "标准差",
        "min": "最小值",
        "max": "最大值",
    })
    desc.insert(0, "变量名称", desc.index)
    return desc.reset_index(drop=True)


def build_abbreviation_table() -> pd.DataFrame:
    """生成 Table A-1：变量简称说明。"""
    return pd.DataFrame(
        VARIABLE_ABBREVIATIONS,
        columns=["变量名称", "英文简称", "英文说明"],
    )


def build_vif_table(df: pd.DataFrame) -> pd.DataFrame:
    """生成附加表：VIF 多重共线性诊断。"""
    X = df[FEATURE_COLS]
    Xs = StandardScaler().fit_transform(X)
    vif = pd.DataFrame({
        "变量名称": X.columns,
        "VIF": [variance_inflation_factor(Xs, i) for i in range(Xs.shape[1])],
    })
    return vif.sort_values("VIF", ascending=False).reset_index(drop=True)


def build_price_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """生成附加表：房价与解释变量相关系数。"""
    corr = df[[PRICE_COL, *FEATURE_COLS]].corr()[PRICE_COL].drop(PRICE_COL).sort_values(ascending=False)
    out = corr.rename(index=ABBREVIATIONS).to_frame("与房价相关系数")
    out.insert(0, "变量名称", [k for k in corr.index])
    out.insert(1, "英文简称", [ABBREVIATIONS[k] for k in corr.index])
    return out.reset_index(drop=True)


def plot_price_trend(df: pd.DataFrame, fig_path: Path) -> None:
    """Figure 1-1：房价变化趋势图。"""
    years = df[YEAR_COL]
    price = df[PRICE_COL]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(years, price, marker="o", linewidth=2)
    ax.set_title(FIGURE_TITLES["fig_1_1"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Average housing price")
    _set_integer_year_ticks(ax, years)
    fig.savefig(fig_path)
    plt.close(fig)


def plot_price_growth(df: pd.DataFrame, fig_path: Path) -> None:
    """Figure 2-1：房价水平与同比增长率。"""
    years = df[YEAR_COL]
    price = df[PRICE_COL]
    yoy = price.pct_change() * 100

    fig, ax1 = plt.subplots(figsize=(10, 5.4))
    ax2 = ax1.twinx()

    ax1.plot(years, price, marker="o", linewidth=2.2)
    ax2.bar(years, yoy, alpha=0.25, width=0.58)

    ax1.set_title(FIGURE_TITLES["fig_2_1"])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Average housing price (yuan/sq.m.)")
    ax2.set_ylabel("YoY growth rate (%)")
    _set_integer_year_ticks(ax1, years)
    ax2.axhline(0, linewidth=0.8)

    fig.savefig(fig_path)
    plt.close(fig)


def plot_index_groups(df: pd.DataFrame, fig_path: Path, columns: list[str], title_key: str) -> None:
    """通用指数图：将多个变量与房价共同转换为2001=100指数。"""
    years = df[YEAR_COL]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for col in columns:
        ax.plot(years, _base_2001_index(df[col]), marker="o", linewidth=2, label=ABBREVIATIONS.get(col, col))

    ax.plot(years, _base_2001_index(df[PRICE_COL]), marker="o", linewidth=2.2, label="Housing price")
    ax.set_title(FIGURE_TITLES[title_key])
    ax.set_xlabel("Year")
    ax.set_ylabel("Index (2001=100)")
    _set_integer_year_ticks(ax, years)
    ax.legend(ncol=2, frameon=False)
    fig.savefig(fig_path)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, fig_path: Path) -> None:
    """Figure 2-5：房价与解释变量相关系数热图。"""
    cols = [PRICE_COL, *FEATURE_COLS]
    corr = df[cols].corr()
    corr.index = [ABBREVIATIONS[c] for c in corr.index]
    corr.columns = [ABBREVIATIONS[c] for c in corr.columns]

    fig, ax = plt.subplots(figsize=(8.6, 7.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(FIGURE_TITLES["fig_2_5"])
    fig.savefig(fig_path)
    plt.close(fig)


def _scatter_with_fit(ax, x, y, x_label: str, corr_text: str | None = None):
    ax.scatter(x, y, s=26)
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(np.min(x), np.max(x), 100)
    yy = coef[0] * xx + coef[1]
    ax.plot(xx, yy, linewidth=1.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Housing price")
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.90, corr_text or f"r = {r:.3f}", transform=ax.transAxes, fontsize=10)


def plot_scatter_panels(df: pd.DataFrame, fig_path: Path) -> None:
    """Figure 2-6：房价与关键变量双变量关系图。"""
    y = df[PRICE_COL].values
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
    axes = axes.ravel()

    selected = [
        ("城镇人均可支配收入（元）", "Urban disposable income"),
        ("人均GDP（元）", "Per capita GDP"),
        ("土地购置费（万元）", "Land acquisition cost"),
        ("三年至五年期贷款利率", "3-5 year loan rate"),
    ]
    for ax, (col, label) in zip(axes, selected):
        _scatter_with_fit(ax, df[col].values, y, label)

    fig.suptitle(FIGURE_TITLES["fig_2_6"], y=1.01)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def generate_descriptive_outputs(df: pd.DataFrame, figure_dir: Path) -> dict[str, pd.DataFrame]:
    """生成描述性统计相关的全部图表与表格。"""
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_price_trend(df, figure_dir / "fig_1_1_price_trend.png")
    plot_price_growth(df, figure_dir / "fig_2_1_price_growth.png")
    plot_index_groups(
        df,
        figure_dir / "fig_2_2_demand_index.png",
        ["人口数量（人）", "人均GDP（元）", "人均个人储蓄存款余额（元）", "城镇人均可支配收入（元）"],
        "fig_2_2",
    )
    plot_index_groups(
        df,
        figure_dir / "fig_2_3_supply_index.png",
        ["土地购置费（万元）", "房地产行业年末从业人数（人）", "完成投资额（万元）", "竣工房屋面积（万平方米）", "开发商市场垄断力指数"],
        "fig_2_3",
    )
    plot_index_groups(
        df,
        figure_dir / "fig_2_4_macro_finance.png",
        ["居民消费价格指数（CPI）", "三年至五年期贷款利率", "开发商市场垄断力指数"],
        "fig_2_4",
    )
    plot_correlation_heatmap(df, figure_dir / "fig_2_5_corr_heatmap.png")
    plot_scatter_panels(df, figure_dir / "fig_2_6_scatter_panels.png")

    return {
        "table_2_1_variable_system": build_variable_system_table(),
        "table_2_2_descriptive_stats": build_descriptive_stats_table(df),
        "table_a_1_variable_abbreviations": build_abbreviation_table(),
        "extra_vif": build_vif_table(df),
        "extra_correlations_with_price": build_price_correlation_table(df),
    }
