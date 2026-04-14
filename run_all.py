"""一键运行主脚本：复现论文图表和模型结果。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, FIGURE_DIRNAME, TABLE_DIRNAME
from src.data_utils import load_data, split_xy
from src.descriptive_analysis import generate_descriptive_outputs
from src.pca_analysis import generate_pca_outputs
from src.modeling import generate_model_outputs
from src.reporting import export_tables, export_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce all figures and tables for the Fuzhou housing price thesis.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to the Excel data file.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    figure_dir = output_dir / FIGURE_DIRNAME
    table_dir = output_dir / TABLE_DIRNAME

    # 1) 读取数据
    df = load_data(args.data)
    years, X, y = split_xy(df)

    # 2) 描述性统计与第2章图表
    desc_tables = generate_descriptive_outputs(df, figure_dir)

    # 3) 主成分分析与第3章图表
    pca_result, pca_tables = generate_pca_outputs(X, figure_dir)

    # 4) PCA-MLR 与 PCA-ANN 建模及第5章图表
    model_results, model_tables = generate_model_outputs(df, figure_dir)

    # 5) 表格汇总导出
    all_tables = {}
    all_tables.update(desc_tables)
    all_tables.update(pca_tables)
    all_tables.update(model_tables)
    export_tables(all_tables, table_dir)

    # 6) 结果摘要导出
    summary = {
        "retained_principal_components": 3,
        "cumulative_variance_explained_first_3": float(pca_result["summary"].loc[2, "累计方差贡献率"]),
        "full_sample_metrics": model_tables["table_5_2_full_sample_metrics"].round(6).to_dict(orient="records"),
        "test_metrics": model_tables["table_5_3_test_metrics"].round(6).to_dict(orient="records"),
        "test_predictions": model_tables["table_5_4_actual_vs_predicted_2020_2021"].round(6).to_dict(orient="records"),
    }
    export_summary(summary, table_dir)

    print("\n已完成全部结果复现。")
    print(f"图形输出目录：{figure_dir}")
    print(f"表格输出目录：{table_dir}")


if __name__ == "__main__":
    main()
