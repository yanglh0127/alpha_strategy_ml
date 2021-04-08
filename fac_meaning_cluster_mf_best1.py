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

# 聚合方式（四）：按逻辑相关性对同一类下所有因子挑选sharpe比率最高的那个

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_20170101-20210228'
fac_meaning = pd.read_excel(data_pat + '/fac_meaning.xlsx', sheet_name='日频资金流向', index_col=0)
fac_perf = pd.read_excel(data_pat + '/perf_summary_eq_tvwap.xlsx', index_col=0)
fac_meaning = fac_meaning[fac_meaning['tag1'] != 'delete']
print(fac_meaning['tag1'].value_counts())
rank_corr = pd.read_csv('E:/Share/FengWang/Alpha/mine/mf_factor/rank_corr.csv', index_col=0)
out_path = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/mf/best1_1'  # 记得修改

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
fac_meaning['cluster_1'] = fac_meaning['tag1'].apply(lambda g: '日间资金流波动' if g in ['开盘资金流的日间波动',
                                                                                  '资金流的日间波动',
                                                                                  '收盘资金流的日间波动',
                                                                                  '主力净流入的绝对值',
                                                                                  '主力净流入的时间趋势绝对值'
                                                                                  ] else ('主力流入流出占比' if g in ['主力流入占比',
                                                                                                              '主力流出占比'
                                                                                                              ] else ('开盘净主动买入行为' if g in ['开盘净主动买入行为',
                                                                                                                                           '开盘和收盘净主动买入之差'
                                                                                                                                           ] else g)))

# 按不同的精细程度记录聚类的各组下因子名、sharp比率、相关性
cluster_h = 'cluster_1'  # 记得修改
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

fac_data = pd.read_pickle(data_pat + '/all_fac_20170101-20210228.pkl')
fac_comb = {}
for tag in cluster_sharp.keys():
    fac_comb[tag] = uc.cs_rank(fac_data[cluster_sharp[tag].idxmax()])
    fac_comb[tag].index = pd.to_datetime(fac_comb[tag].index)
f = open(out_path + '/fac.pkl', 'wb')
pickle.dump(fac_comb, f, -1)
f.close()

# 新聚合因子之间的相关性
co_rank = cal_factor_corr(fac_comb, out_path)
print(co_rank)
"""
# 把聚合因子的表现结果汇总
type = 'best1_1'  # 记得修改
perf_path = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/mf/' + str(type) + '/eq_tvwap'
results_perf = {}
results_hperf = {}
results_to = {}
comb_group = [i for i in os.listdir(perf_path) if os.path.isdir(os.path.join(perf_path, i))]
for cg in comb_group:
    nn_dir = os.path.join(perf_path, cg)
    for j in os.listdir(nn_dir):
        if j[-3:] == 'pkl':
            result = pd.read_pickle(os.path.join(nn_dir, j))
            results_perf[type + '_' + cg] = result['perf']
            results_hperf[type + '_' + cg] = result['hedged_perf']
            results_to[type + '_' + cg] = result['turnover_series'].mean()

perf = pd.concat(results_perf, axis=1)
hperf = pd.concat(results_hperf, axis=1)
hperf.index = 'H_' + hperf.index
to = pd.DataFrame.from_dict(results_to, orient='index')
to.columns = ['turnover']
perf_summary = pd.concat([perf, hperf])
perf_summary = pd.concat([perf_summary.T, to], axis=1)
perf_summary.to_csv(perf_path + '/tp.csv', encoding='utf_8_sig')
"""
