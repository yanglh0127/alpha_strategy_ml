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

# 聚合方式（三）：先选出sharp大于0的因子，按逻辑相关性对同一类下的因子按sharpe比率进行加权聚合

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_20170101-20210228'  # 这边路径记得改
fac_meaning = pd.read_excel(data_pat + '/fac_meaning.xlsx', sheet_name='高频资金流向', index_col=0)  # 记得修改
fac_perf = pd.read_excel(data_pat + '/perf_summary_eq_tvwap.xlsx', index_col=0)  # 记得修改
print(fac_meaning['tag1'].unique())
rank_corr = pd.read_csv('E:/Share/FengWang/Alpha/mine/hfmf_factor/rank_corr.csv', index_col=0)  # 记得修改
out_path = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/hfmf/sharpe_weight'  # 记得修改

# 只选择夏普比率排行前50%
fac_meaning = fac_meaning[fac_meaning['sharp_ratio'] > 0]
print(fac_meaning['tag1'].unique())

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

# 按tag1的取值分类，按分类的细致程度分为cluster_0, cluster_1, cluster_2
fac_meaning['cluster_0'] = fac_meaning['tag1']

# 按不同的精细程度记录聚类的各组下因子名、sharp比率、相关性
cluster_h = 'cluster_0'  # 记得修改
cluster_corr = {}
cluster_sharp = {}
for tag in list(fac_meaning[cluster_h].unique()):
    temp = fac_meaning[fac_meaning[cluster_h] == tag].index.tolist()
    temp_name = [i[15:-3] for i in temp]
    print(tag, len(temp))
    co = rank_corr.loc[temp_name, temp_name]
    co1 = co.reindex(co.columns)  # 调整顺序，化为对称阵
    cluster_corr[tag] = co1.mask(co1.isna(), co1.T)
    sharp = fac_meaning.loc[temp, 'sharp_ratio']
    sharp.index = [i[15:-3] for i in sharp.index.tolist()]
    cluster_sharp[tag] = sharp
    fac_perf.loc[temp, :].to_csv(out_path + '/' + str(tag) + '.csv')

# 因子聚合
fac_data = pd.read_pickle(data_pat + '/all_fac_20170101-20210228.pkl')  # 记得修改
fac_comb = {}
for tag in cluster_sharp.keys():
    temp = {}
    for i in cluster_sharp[tag].index.tolist():
        temp[i] = uc.cs_rank(fac_data[i]) * cluster_sharp[tag][i]
    comb = pd.concat(temp.values())
    fac_comb[tag] = comb.groupby(comb.index).mean()
    fac_comb[tag].index = pd.to_datetime(fac_comb[tag].index)
f = open(out_path + '/fac.pkl', 'wb')
pickle.dump(fac_comb, f, -1)
f.close()

# 新聚合因子之间的相关性
co_rank = cal_factor_corr(fac_comb, out_path)
print(co_rank)
