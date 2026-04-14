# 福州市商品房价格影响因素分析与房价预测：可复现实证代码包

本代码包用于**完整复现论文中的主要图表、主成分分析结果、PCA-MLR 与 PCA-ANN 模型拟合结果以及 2020–2021 年类预测检验结果**。

## 1. 代码包内容

```text
Fuzhou_Housing_Code_Package/
├─ data/
│  └─ data.xlsx                         # 论文使用的原始数据
├─ docs/
│  ├─ 代码使用说明.md                    # 详细使用说明
│  └─ 输出文件对照表.md                  # 图表编号与文件名对照
├─ notebooks/
│  └─ reproduce_all_results.ipynb      # Jupyter 复现版
├─ outputs/
│  ├─ figures/                         # 运行后生成的全部图形
│  └─ tables/                          # 运行后生成的全部表格
├─ src/
│  ├─ config.py                        # 变量定义、题注、模型参数
│  ├─ data_utils.py                    # 数据读取与校验
│  ├─ descriptive_analysis.py          # 描述性统计与第2章图表
│  ├─ pca_analysis.py                  # 主成分分析与第3章图表
│  ├─ modeling.py                      # PCA-MLR / PCA-ANN 拟合与第5章图表
│  └─ reporting.py                     # Excel/CSV/JSON 导出工具
├─ run_all.py                          # 一键运行主脚本
└─ requirements.txt                    # Python 依赖
```

## 2. 运行环境

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

## 3. 一键复现全部结果

在代码包根目录运行：

```bash
python run_all.py
```

Windows 也可以双击运行 `run_all.bat`；Linux / macOS 可运行 `bash run_all.sh`。

如果你要指定自己的数据文件与输出目录：

```bash
python run_all.py --data data/data.xlsx --output outputs
```

## 4. 运行后将自动生成的内容

### 图形
- Figure 1-1: 房价变化趋势图
- Figure 2-1 ～ Figure 2-6: 描述性统计与关系图
- Figure 3-1 ～ Figure 3-2: PCA 碎石图与载荷热图
- Figure 5-1 ～ Figure 5-2: 模型拟合与类预测检验图

### 表格
- Table 2-1: 变量体系与理论预期
- Table 2-2: 描述性统计
- Table 3-1: 主成分特征值与方差贡献率
- Table 3-2: 前3个主成分载荷矩阵
- Table 5-1: PCA-MLR 模型系数估计结果（训练集）
- Table 5-2: 两类模型全样本拟合效果比较
- Table 5-3: 两类模型类预测检验效果比较（2020–2021）
- Table 5-4: 2020–2021 年实际值与预测值比较
- Table A-1: 变量简称说明
- Table A-2: 核心建模参数

另附加导出：
- extra_vif.csv：多重共线性诊断
- extra_correlations_with_price.csv：房价与各解释变量相关系数
- summary_results.json：关键结果汇总
- thesis_tables.xlsx：全部表格汇总工作簿

## 5. 说明

1. 代码中的随机种子、主成分个数、神经网络结构均已固定，以保证结果可复现。
2. 所有代码文件都加入了注释，便于后续修改或扩展。
3. 若替换为其他城市的数据，只需保持列名一致即可复用本代码框架。
