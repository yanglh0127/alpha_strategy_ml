
## 文件路径
# path_data_asharescore = r'E:\YuanyuanShi\Python\Alpha\data\vp10_new151_alex_novp15_375_SH.csv'
path_results = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster/fac_addfunda/hong_add_cost/eq_tvwap'  # 记得修改

# 回测参数
PARAMS_BT = {
    "buy_commission": 0.0002,  # 买入成本，选择用均价交易时，可以不增加交易冲击，交易冲击默认0.001  # 原本为万分之二
    "sell_commission": 0.0002,  # 卖出成本，选择用均价交易时，可以不考虑交易冲击  # 原本为万分之二
    "tax_ratio": 0.001,  # 印花税
    "capital": 10000000,  # 虚拟资本, 没考虑股指期货所需要的资金
    "risk_free_rate": 0.0,  # 无风险利率 (not percentage)
    "benchmark": "000905",  # 基准指数
}

col_industries_name = 'industry_component_SW1'
col_index_component_weight = 'index_weight_000905'
cols_eodprices = ['st_or_not', 'suspend', 'stock_tcap', 'stock_mcap']
fields_daily = ['stock_close', 'stock_open', 'stock_lclose', 'stock_matiply_ratio',
                'suspend', 'st_or_not', 'stock_high', 'stock_low', 'maxupordown']
# ------------------------------------------------------------
##### Portfolio Management
# 组合优化参数（具体含义参见pm.Portfolio_Management的文档）
PARAMS_PM = {
    "num_holdings": 100,  # 记得修改
    "Epsilon": 0.005,
    "industry_neutral": None,
    "industry_expourse": None,
    "style_neutral": None,
    "style_expourse": None,
    "upper_bound": None,
    "lower_bound": None,
    "to_bound": 0.2,  # 记得修改
    "te_bound": None,
    "vol_bound": None,
    "benchmark": "000905.SH",
    "constituents_only": False,
    "signal_direction": 1,
    "update": False,
    "disp_info": False,
    "ratio_max_holdings": 1.5,
    "constraints_relax": {
        "step": {"to": 0.005, "TE": 0.01, "exposure": 0.0001, "upper_bound": 0.0005},
        "limit": {"to": 0.8, "TE": 0.3, "exposure": 1.0, "upper_bound": 0.05},
    },
    "convergency_relax": {"multiple_max_iters": 10, "tol2stop": 1e-4},
    "raise_SolverError": False,
}

industry_exposure = {'D_801780': 1, 'D_801180': 1, 'D_801150': 1, 'D_801160': 1, 'D_801200': 1, 'D_801890': 1,
                     'D_801230': 1, 'D_801720': 1, 'D_801710': 1, 'D_801110': 1, 'D_801880': 1, 'D_801080': 1,
                     'D_801140': 1, 'D_801770': 1, 'D_801750': 1, 'D_801760': 1, 'D_801790': 1, 'D_801010': 1,
                     'D_801030': 1, 'D_801050': 1, 'D_801170': 1, 'D_801730': 1, 'D_801210': 1, 'D_801740': 1,
                     'D_801020': 1, 'D_801120': 1, 'D_801130': 1, 'D_801040': 1}

style_exposure = {
    "size": 1,
    "volatility": 1,
    "growth": 1,
    "valuation": 1,
    "earning": 1,
    "momentum": 1,
    "liquidity": 1,
    "beta": 1,
    "leverage": 1}  # 可以选择需要约束的风格因子列

style_neutral = ["size", "volatility"]  # 设置需要中性化的风格变量

# 风格约束的分组情况，默认平均分成10组
style_categorical = {
    "size": 10,
    "volatility": 10,
    "growth": 10,
    "valuation": 10,
    "earning": 10,
    "momentum": 10,
    "liquidity": 10,
    "beta": 10,
    "leverage": 10

}
