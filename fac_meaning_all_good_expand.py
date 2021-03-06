import pandas as pd
import os
import h5py
import json
from ft_platform.utils import utils_calculation as uc
import itertools
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt
import pickle
from ft_platform.factor_process import fetch
from utils_func import query_data

# 在扩充的因子池基础上，把各个类下表现好的聚合因子整理到一起

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'

file_list = ['all_eq', '50%_eq', '30%_eq', '15%_eq', 'sharpe_weight', 'best1']
fac_chosen_vphfvp_old = ['sharpe_weight_反转因子相关_vp', 'sharpe_weight_1_高频量价相关性_hfvp']

fac_chosen_vphfvp_new = ['15%_eq_1_反转因子相关_hfvp', 'sharpe_weight_1_情绪因子_vp', '50%_eq_1_收益率和波动率的相关性_vp',
                         'sharpe_weight_1_收盘行为异常_hfvp', '15%_eq_1_日内不同时段成交量差异_hfvp', '15%_eq_1_日内成交额分布的稳定性_hfvp',
                         'sharpe_weight_1_日内成交额的自相关_hfvp', 'sharpe_weight_1_日内收益率的分布_hfvp', '15%_eq_1_日间成交量(额)的波动率_vp',
                         'sharpe_weight_1_流动性因子改进_hfvp', '50%_eq_1_流动性因子相关_vp', 'sharpe_weight_1_量价相关性_vp',
                         'all_eq_1_隔夜(或上午)和下午收益率差异_hfvp', '50%_eq_1_高频收益率为正和负时的波动率差异_hfvp', '15%_eq_1_高频贝塔_hfvp']

fac_chosen_hfmf = ['50%_eq_1_大单行为_hfmf', 'all_eq_1_反转因子改进_hfmf', '50%_eq_1_高频资金流分布_hfmf']

fac_chosen_mf = ['best1_1_反转因子改进_mf', 'best1_1_价格和资金流向的相关性_mf', 'best1_1_开盘净主动买入行为_mf',
                 'sharpe_weight_1_日间资金流波动_mf', 'best1_1_收盘主力净流入行为_mf', 'best1_1_主力单数行为_mf',
                 'best1_1_主力净流入占比的偏度_mf', 'sharpe_weight_1_主力流入流出占比_mf']

fac_old = pd.read_pickle(data_pat + '/fac_last.pkl')
fac_old = {k: v for k, v in fac_old.items() if k in (fac_chosen_mf + fac_chosen_hfmf + fac_chosen_vphfvp_old)}

fac_new = {}
for file_name in file_list:
    temp = pd.read_pickle(data_pat + '/fac_expand/' + file_name + '/fac.pkl')
    fac_new = dict(fac_new, **temp)
fac_new = {k: v for k, v in fac_new.items() if k in fac_chosen_vphfvp_new}
fac_all = dict(fac_old, **fac_new)
f = open(data_pat + '/fac_expand/all_cluster/fac_last.pkl', 'wb')
pickle.dump(fac_all, f, -1)
f.close()

# 基础函数，计算因子之间的相关系数
def cal_factor_corr(fac_dict, pat_str):
    if not os.path.exists(pat_str):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(pat_str)
    total_data = pd.concat(fac_dict.values(), keys=fac_dict.keys())
    total_data = total_data.reset_index().set_index('level_1')
    corank_total = total_data.groupby(total_data.index).apply(lambda g: g.set_index('level_0').T.corr('spearman'))
    co_rank = corank_total.groupby(corank_total.index.get_level_values(1)).mean()
    co_rank = co_rank.reindex(co_rank.columns)  # 调整顺序，化为对称阵
    co_rank.to_csv(pat_str + "/rank_corr.csv", index=True, encoding='utf_8_sig')
    return co_rank

# 计算各类因子之间的相关性
co_rank = cal_factor_corr(fac_all, data_pat + '/fac_expand/all_cluster')
print(co_rank)
