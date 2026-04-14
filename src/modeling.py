"""PCA-MLR 与 PCA-ANN 建模、评价和第5章图表生成。"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    YEAR_COL, PRICE_COL, FEATURE_COLS, PCA_COMPONENTS, ANN_PARAMS,
    TRAIN_END_YEAR, TEST_START_YEAR, FIGURE_TITLES,
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.bbox"] = "tight"


def metrics(y_true, y_pred) -> dict[str, float]:
    """计算 RMSE、MAE、MAPE 和 R²。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_parameter_table() -> pd.DataFrame:
    """生成 Table A-2：核心建模参数。"""
    rows = [
        ("主成分个数", str(PCA_COMPONENTS)),
        ("训练集时间范围", f"2001-{TRAIN_END_YEAR}"),
        ("检验集时间范围", f"{TEST_START_YEAR}-2021"),
    ]
    for k, v in ANN_PARAMS.items():
        rows.append((f"ANN 参数：{k}", str(v)))
    return pd.DataFrame(rows, columns=["参数项", "参数值"])


def fit_full_sample_models(df: pd.DataFrame) -> dict:
    """全样本拟合：比较 PCA-MLR 与 PCA-ANN 的样本内表现。"""
    X = df[FEATURE_COLS]
    y = df[PRICE_COL].values

    lr_full = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("lr", LinearRegression()),
    ])
    ann_full = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("mlp", MLPRegressor(**ANN_PARAMS)),
    ])
    lr_full.fit(X, y)
    ann_full.fit(X, y)

    pred_lr = lr_full.predict(X)
    pred_ann = ann_full.predict(X)

    metrics_table = pd.DataFrame([
        {"模型": "PCA-MLR", **metrics(y, pred_lr)},
        {"模型": "PCA-ANN", **metrics(y, pred_ann)},
    ])

    fitted_df = pd.DataFrame({
        YEAR_COL: df[YEAR_COL],
        "Actual": y,
        "PCA-MLR": pred_lr,
        "PCA-ANN": pred_ann,
    })

    return {
        "lr_model": lr_full,
        "ann_model": ann_full,
        "metrics_table": metrics_table,
        "fitted_df": fitted_df,
    }


def fit_train_test_models(df: pd.DataFrame) -> dict:
    """类预测检验：2001-2019 训练，2020-2021 检验。"""
    train = df[df[YEAR_COL] <= TRAIN_END_YEAR].copy()
    test = df[df[YEAR_COL] >= TEST_START_YEAR].copy()

    X_train = train[FEATURE_COLS]
    y_train = train[PRICE_COL].values
    X_test = test[FEATURE_COLS]
    y_test = test[PRICE_COL].values

    scaler_train = StandardScaler().fit(X_train)
    X_train_s = scaler_train.transform(X_train)
    X_test_s = scaler_train.transform(X_test)

    pca_train = PCA(n_components=PCA_COMPONENTS).fit(X_train_s)
    Z_train = pca_train.transform(X_train_s)
    Z_test = pca_train.transform(X_test_s)

    # PCA-MLR 使用 statsmodels 方便导出系数表
    ols = sm.OLS(y_train, sm.add_constant(Z_train)).fit()
    pred_lr_train = ols.predict(sm.add_constant(Z_train))
    pred_lr_test = ols.predict(sm.add_constant(Z_test))

    ann = MLPRegressor(**ANN_PARAMS)
    ann.fit(Z_train, y_train)
    pred_ann_train = ann.predict(Z_train)
    pred_ann_test = ann.predict(Z_test)

    coef_df = pd.DataFrame({
        "变量": ["const", "Z1", "Z2", "Z3"],
        "系数": ols.params,
        "标准误": ols.bse,
        "t值": ols.tvalues,
        "P值": ols.pvalues,
        "95%下限": ols.conf_int()[:, 0],
        "95%上限": ols.conf_int()[:, 1],
    })

    full_pred_table = pd.DataFrame({
        YEAR_COL: test[YEAR_COL],
        "实际值": y_test,
        "PCA-MLR预测值": pred_lr_test,
        "PCA-ANN预测值": pred_ann_test,
    })

    train_metrics = pd.DataFrame([
        {"模型": "PCA-MLR", **metrics(y_train, pred_lr_train)},
        {"模型": "PCA-ANN", **metrics(y_train, pred_ann_train)},
    ])
    test_metrics = pd.DataFrame([
        {k: v for k, v in {"模型": "PCA-MLR", **metrics(y_test, pred_lr_test)}.items() if k in ["模型", "RMSE", "MAE", "MAPE", "R2"]},
        {k: v for k, v in {"模型": "PCA-ANN", **metrics(y_test, pred_ann_test)}.items() if k in ["模型", "RMSE", "MAE", "MAPE", "R2"]},
    ])

    return {
        "train_df": train,
        "test_df": test,
        "ols": ols,
        "ann": ann,
        "coef_df": coef_df,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_pred_df": full_pred_table,
        "train_predictions": {
            "PCA-MLR": pred_lr_train,
            "PCA-ANN": pred_ann_train,
        },
        "test_predictions": {
            "PCA-MLR": pred_lr_test,
            "PCA-ANN": pred_ann_test,
        },
    }


def plot_full_sample_fit(fitted_df: pd.DataFrame, fig_path: Path) -> None:
    """Figure 5-1：全样本实际值与拟合值比较。"""
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.plot(fitted_df[YEAR_COL], fitted_df["Actual"], marker="o", linewidth=2.2, label="Actual")
    ax.plot(fitted_df[YEAR_COL], fitted_df["PCA-MLR"], marker="s", linewidth=1.8, label="PCA-MLR")
    ax.plot(fitted_df[YEAR_COL], fitted_df["PCA-ANN"], marker="^", linewidth=1.8, label="PCA-ANN")
    ax.set_title(FIGURE_TITLES["fig_5_1"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Housing price")
    ax.set_xticks(list(fitted_df[YEAR_COL]))
    ax.legend(frameon=False)
    fig.savefig(fig_path)
    plt.close(fig)


def plot_test_compare(test_pred_df: pd.DataFrame, fig_path: Path) -> None:
    """Figure 5-2：2020-2021 类预测检验图。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(test_pred_df[YEAR_COL], test_pred_df["实际值"], marker="o", linewidth=2.2, label="Actual")
    ax.plot(test_pred_df[YEAR_COL], test_pred_df["PCA-MLR预测值"], marker="s", linewidth=1.8, label="PCA-MLR")
    ax.plot(test_pred_df[YEAR_COL], test_pred_df["PCA-ANN预测值"], marker="^", linewidth=1.8, label="PCA-ANN")
    ax.set_title(FIGURE_TITLES["fig_5_2"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Housing price")
    ax.set_xticks(list(test_pred_df[YEAR_COL]))
    ax.legend(frameon=False)
    fig.savefig(fig_path)
    plt.close(fig)


def generate_model_outputs(df: pd.DataFrame, figure_dir: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    """生成第5章图表及相关结果对象。"""
    figure_dir.mkdir(parents=True, exist_ok=True)
    full_res = fit_full_sample_models(df)
    test_res = fit_train_test_models(df)

    plot_full_sample_fit(full_res["fitted_df"], figure_dir / "fig_5_1_full_sample_fit.png")
    plot_test_compare(test_res["test_pred_df"], figure_dir / "fig_5_2_test_compare.png")

    tables = {
        "table_5_1_pca_mlr_coefficients_train": test_res["coef_df"],
        "table_5_2_full_sample_metrics": full_res["metrics_table"],
        "table_5_3_test_metrics": test_res["test_metrics"],
        "table_5_4_actual_vs_predicted_2020_2021": test_res["test_pred_df"],
        "table_a_2_model_parameters": build_parameter_table(),
    }
    return {"full": full_res, "test": test_res}, tables
