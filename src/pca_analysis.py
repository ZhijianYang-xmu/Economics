"""主成分分析与第3章图表生成。"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLS, FIGURE_TITLES, PCA_COMPONENTS, ABBREVIATIONS

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.bbox"] = "tight"


def run_pca(X: pd.DataFrame) -> dict:
    """对特征矩阵执行标准化和 PCA。"""
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    pca = PCA().fit(X_scaled)

    summary = pd.DataFrame({
        "主成分": [f"PC{i+1}" for i in range(X.shape[1])],
        "特征值": pca.explained_variance_,
        "方差贡献率": pca.explained_variance_ratio_,
        "累计方差贡献率": pca.explained_variance_ratio_.cumsum(),
    })

    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=FEATURE_COLS,
        columns=[f"PC{i+1}" for i in range(X.shape[1])],
    )
    return {
        "scaler": scaler,
        "pca": pca,
        "summary": summary,
        "loadings": loadings,
        "n_components": PCA_COMPONENTS,
    }


def plot_scree(pca_summary: pd.DataFrame, fig_path: Path) -> None:
    """Figure 3-1：碎石图。"""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = np.arange(1, len(pca_summary) + 1)
    ax.plot(xs, pca_summary["特征值"], marker="o", linewidth=2)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_title(FIGURE_TITLES["fig_3_1"])
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Eigenvalue")
    ax.set_xticks(xs)
    fig.savefig(fig_path)
    plt.close(fig)


def plot_loading_heatmap(loadings: pd.DataFrame, fig_path: Path) -> None:
    """Figure 3-2：前3个主成分载荷热图。"""
    heat = loadings.iloc[:, :3].copy()
    heat.index = [ABBREVIATIONS[i] for i in heat.index]
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(FIGURE_TITLES["fig_3_2"])
    fig.savefig(fig_path)
    plt.close(fig)


def generate_pca_outputs(X: pd.DataFrame, figure_dir: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    """生成 PCA 分析图表与结果对象。"""
    figure_dir.mkdir(parents=True, exist_ok=True)
    pca_result = run_pca(X)
    plot_scree(pca_result["summary"], figure_dir / "fig_3_1_scree_plot.png")
    plot_loading_heatmap(pca_result["loadings"], figure_dir / "fig_3_2_loading_heatmap.png")

    tables = {
        "table_3_1_pca_summary": pca_result["summary"],
        "table_3_2_pca_loadings": pca_result["loadings"].reset_index().rename(columns={"index": "变量名称"}),
    }
    return pca_result, tables
