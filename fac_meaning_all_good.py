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

# 把各个类下表现好的聚合因子整理到一起

group = 'vp'  # 记得修改
file_list = ['all_eq', '50%_eq', 'sharpe_weight', 'best_1']  # 记得修改
fac_chosen = ['sharpe_weight_反转因子相关', 'sharpe_weight_量价相关性', 'all_eq_流动性因子相关',
              'sharpe_weight_情绪因子', 'best_1_日间成交量(额)的波动率', '50%_eq_收益率和波动率的相关性']  # 记得修改
data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning'

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

fac_comb = {}
for file_name in file_list:
    temp = pd.read_pickle(data_pat + '/' + group + '/' + file_name + '/fac.pkl')
    for fac_name in temp.keys():
        fac_comb[(file_name + '_' + fac_name)] = temp[fac_name]
fac_comb = {k: v for k, v in fac_comb.items() if k in fac_chosen}
f = open(data_pat + '/' + group + '/fac_last.pkl', 'wb')
pickle.dump(fac_comb, f, -1)
f.close()

# 新聚合因子之间的相关性
co_rank = cal_factor_corr(fac_comb, data_pat + '/' + group)
print(co_rank)