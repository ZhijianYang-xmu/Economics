"""表格与摘要信息导出。"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def export_tables(tables: dict[str, pd.DataFrame], table_dir: Path) -> None:
    """将所有表格分别导出为 CSV，并汇总到一个 Excel 工作簿。"""
    table_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = table_dir / "thesis_tables.xlsx"

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for name, df in tables.items():
            csv_path = table_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            sheet_name = name[:31]
            df.to_excel(writer, index=False, sheet_name=sheet_name)


def export_summary(summary: dict, table_dir: Path) -> None:
    """导出关键结果摘要为 JSON。"""
    out_path = table_dir / "summary_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
