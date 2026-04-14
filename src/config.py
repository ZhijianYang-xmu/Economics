"""配置文件：集中定义变量、图表题注、模型参数。"""

from pathlib import Path

# 根目录路径
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PACKAGE_ROOT / "data" / "data.xlsx"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs"
FIGURE_DIRNAME = "figures"
TABLE_DIRNAME = "tables"

# 原始变量名
YEAR_COL = "年份"
PRICE_COL = "商品房平均销售价格（元/平方米）"
FEATURE_COLS = [
    "人口数量（人）",
    "人均GDP（元）",
    "人均个人储蓄存款余额（元）",
    "城镇人均可支配收入（元）",
    "土地购置费（万元）",
    "房地产行业年末从业人数（人）",
    "完成投资额（万元）",
    "竣工房屋面积（万平方米）",
    "开发商市场垄断力指数",
    "居民消费价格指数（CPI）",
    "三年至五年期贷款利率",
]

# 英文简称，用于热图、图例和附录
ABBREVIATIONS = {
    PRICE_COL: "Price",
    "人口数量（人）": "Population",
    "人均GDP（元）": "PerCapGDP",
    "人均个人储蓄存款余额（元）": "Savings",
    "城镇人均可支配收入（元）": "Income",
    "土地购置费（万元）": "LandCost",
    "房地产行业年末从业人数（人）": "Employment",
    "完成投资额（万元）": "Investment",
    "竣工房屋面积（万平方米）": "CompletedArea",
    "开发商市场垄断力指数": "Monopoly",
    "居民消费价格指数（CPI）": "CPI",
    "三年至五年期贷款利率": "LoanRate",
}

# 变量分组与理论预期
VARIABLE_SYSTEM = [
    ("需求因素", "人口数量（人）", "Population", "+", "人口集聚提高住房需求，通常推动房价上升"),
    ("需求因素", "人均GDP（元）", "PerCapGDP", "+", "经济发展提高居民购买能力与住房需求"),
    ("需求因素", "人均个人储蓄存款余额（元）", "Savings", "+", "储蓄增长提升首付能力与改善型住房需求"),
    ("需求因素", "城镇人均可支配收入（元）", "Income", "+", "收入提高增强实际支付能力"),
    ("供给因素", "土地购置费（万元）", "LandCost", "+", "土地成本上升会抬高开发成本并向房价传导"),
    ("供给因素", "房地产行业年末从业人数（人）", "Employment", "+", "开发活跃度上升通常与房地产景气扩张同步"),
    ("供给因素", "完成投资额（万元）", "Investment", "+/-", "投资扩张既可能体现需求旺盛，也可能增加供给，作用可能双向"),
    ("供给因素", "竣工房屋面积（万平方米）", "CompletedArea", "-", "竣工面积增加会缓解供给约束，对房价有抑制作用"),
    ("其他因素", "开发商市场垄断力指数", "Monopoly", "+", "垄断程度较高可能增强开发商定价能力"),
    ("其他因素", "居民消费价格指数（CPI）", "CPI", "+", "通货膨胀会通过成本和资产保值动机影响房价"),
    ("其他因素", "三年至五年期贷款利率", "LoanRate", "-", "利率上升提高按揭融资成本，抑制住房需求"),
]

# 附录：变量简称表
VARIABLE_ABBREVIATIONS = [
    (PRICE_COL, "Price", "Average commercial housing selling price"),
    ("人口数量（人）", "Population", "Population"),
    ("人均GDP（元）", "PerCapGDP", "Per capita GDP"),
    ("人均个人储蓄存款余额（元）", "Savings", "Per capita personal savings deposits"),
    ("城镇人均可支配收入（元）", "Income", "Urban per capita disposable income"),
    ("土地购置费（万元）", "LandCost", "Land acquisition cost"),
    ("房地产行业年末从业人数（人）", "Employment", "Year-end employment in the real estate industry"),
    ("完成投资额（万元）", "Investment", "Completed real estate investment"),
    ("竣工房屋面积（万平方米）", "CompletedArea", "Completed housing area"),
    ("开发商市场垄断力指数", "Monopoly", "Developer market monopoly index"),
    ("居民消费价格指数（CPI）", "CPI", "Consumer price index"),
    ("三年至五年期贷款利率", "LoanRate", "3-5 year loan interest rate"),
]

# 模型参数
PCA_COMPONENTS = 3
ANN_PARAMS = {
    "hidden_layer_sizes": (3,),
    "activation": "tanh",
    "solver": "lbfgs",
    "alpha": 0.01,
    "max_iter": 2000,
    "random_state": 42,
}
TRAIN_END_YEAR = 2019
TEST_START_YEAR = 2020

# 绘图题目（图内部）
FIGURE_TITLES = {
    "fig_1_1": "Fuzhou commercial housing price trend (2001-2021)",
    "fig_2_1": "Housing price level and growth rate in Fuzhou (2001-2021)",
    "fig_2_2": "Demand-side indicators and housing price in index form",
    "fig_2_3": "Supply-side indicators and housing price in index form",
    "fig_2_4": "Macro-financial indicators and housing price in index form",
    "fig_2_5": "Correlation heatmap of housing price and explanatory variables",
    "fig_2_6": "Selected bivariate relationships between housing price and key factors",
    "fig_3_1": "Scree plot of principal components",
    "fig_3_2": "Heatmap of loadings for the first three principal components",
    "fig_5_1": "Comparison of actual and fitted housing prices for the full sample",
    "fig_5_2": "Pseudo out-of-sample forecast comparison (2020-2021)",
}
