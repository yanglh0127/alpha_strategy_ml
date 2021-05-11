from ft_platform import fetch_data
from ft_platform.utils import utils_calculation as uc
import pandas as pd
import pickle
from utils_func import query_data
from ft_platform.factor_process import fetch
import json
from copy import deepcopy
import numpy as np
import time
import os
from alpha_portfolios import backtest_dailyreturns as bd

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'

begin = '2017-01-01'  # 记得修改
end = '2020-08-31'  # 记得修改

# 读取因子数据
fac_old = pd.read_pickle(data_pat + '/fac_last.pkl')
fac_fundamental = {}
pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/fundamental/'
for way in ['all_eq', '50%_eq', 'sharpe_weight']:
    for j in os.listdir(pat + way):
        if j[-3:] == 'pkl':
            temp = pd.read_pickle(pat + way + '/' + j)
            fac_fundamental = dict(fac_fundamental, **temp)
fac_fundamental = {k: v for k, v in fac_fundamental.items() if k in ['50%_eq_fundamental_growth',
                                                                     '50%_eq_fundamental_earning',
                                                                     'sharpe_weight_fundamental_valuation']}
fac_org = dict(fac_old, **fac_fundamental)
trade_days = query_data.get_trade_days('d', from_trade_day=begin, to_trade_day=end)
fac_org = {k: v.loc[trade_days] for k, v in fac_org.items()}

# 取负号，扩充因子
fac_data = {}
for k, v in fac_org.items():
    fac_data[k] = v
    fac_data[k + '_neg'] = -v

# 在原来的最优基础上遍历地添加一个
def add_fac(base_comb, base_fac, wait_delete, not_com):
    fac_comb = {}
    for fac_add in wait_delete:
        com_name = '(' + base_comb + ',' + fac_add + ')'
        if com_name != not_com:
            temp = {k: v for k, v in base_fac.items()}
            temp[fac_add] = uc.cs_rank(fac_data[fac_add])
            comb = pd.concat(temp.values())
            fac_comb[com_name] = comb.groupby(comb.index).mean()
            fac_comb[com_name].index = pd.to_datetime(fac_comb[com_name].index)
    return fac_comb


# 组合权重设置（二）：多头等权重
def get_equal_weight_individual(signal=pd.DataFrame(), start_date='2017-01-01', end_date='2020-08-31'):
    signal = signal[(signal.index >= start_date) & (signal.index <= end_date)]
    weight = (uc.cs_rank(signal) >= 0.9).astype(int)
    weight = weight.div(weight.sum(axis=1), axis=0)
    weight = weight.where(weight > 0)
    weight = weight.dropna(axis=1, how='all')
    return weight


# 策略回测
def strategy_backtest_individual(opm_weight, out_file):
    df_port = opm_weight
    df_port = (df_port.T / df_port.T.sum()).T
    df = df_port.stack().reset_index()
    df.columns = ['trade_date', 'code', 'weight']
    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)
    init_account = 10000000.0
    df['amt'] = init_account * df['weight']
    df = df[df['amt'] > 0]
    portfolios = df.copy()
    pro_bt = bd.BackTest(portfolios, fields_trade_prices_buy='stock_twap_0930_1030', fields_trade_prices_sell='stock_twap_0930_1030')  # 记得修改
    pro_bt.run(filename=out_file)


# 回测因子的策略效果
def test_fac(fac_dict):
    for fac in fac_dict.keys():
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        wt = get_equal_weight_individual(fac_dict[fac], begin, end)
        strategy_backtest_individual(wt, fac)
    # 读取因子表现数据
    perf_path = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster/fac_addfunda/hong_add_cost_neg/eq_tvwap'
    results_perf = {}
    results_hperf = {}
    results_to = {}
    for cg in fac_dict.keys():
        nn_dir = os.path.join(perf_path, cg)
        for j in os.listdir(nn_dir):
            if j[-3:] == 'pkl':
                result = pd.read_pickle(os.path.join(nn_dir, j))
                results_perf[cg] = result['perf']
                results_hperf[cg] = result['hedged_perf']
                results_to[cg] = result['turnover_series'].mean()

    perf = pd.concat(results_perf, axis=1)
    hperf = pd.concat(results_hperf, axis=1)
    hperf.index = 'H_' + hperf.index
    to = pd.DataFrame.from_dict(results_to, orient='index')
    to.columns = ['turnover']
    perf_summary = pd.concat([perf, hperf])
    perf_summary = pd.concat([perf_summary.T, to], axis=1)
    perf_summary = perf_summary.sort_values(by='sharp_ratio', axis=0, ascending=False)
    new_com = perf_summary.index[0]
    new_sharp = perf_summary.loc[new_com, 'sharp_ratio']
    print("增加一个因子后的最优组合 ", new_com, "增加一个因子后的最优夏普 ", new_sharp)
    return new_sharp, new_com

fac_info = pd.read_excel(data_pat + '/fac_addfunda/all_addfunda.xlsx', sheet_name='各类聚合因子的表现', index_col=0)
# 初始化
wait_del = list(fac_data.keys())
base_com = fac_info.index.to_list()[0]
base_fa = {}
base_fa[base_com] = uc.cs_rank(fac_data[base_com])
base_sharpe = fac_info.loc[base_com, 'sharp_ratio']
wait_del.remove(base_com)
not_comb = '(' + base_com + ',' + base_com + '_neg)'  # 不能出现的组合

while(len(wait_del) > 0):
    print("当前最优因子组合: ", base_com, "当前最优夏普比率: ", base_sharpe)
    # 在当前最优的基础上遍历添加一个因子
    fac_new = add_fac(base_com, base_fa, wait_del, not_comb)
    # 回测因子的策略效果
    new_sharp, new_com = test_fac(fac_new)
    if new_sharp > base_sharpe:
        base_com = new_com
        base_sharpe = new_sharp
        rem = base_com.split(',')[-1][:-1]  # list中要去除的
        base_fa[rem] = uc.cs_rank(fac_data[rem])
        print("移除 ", rem)
        wait_del.remove(rem)
    else:
        break
