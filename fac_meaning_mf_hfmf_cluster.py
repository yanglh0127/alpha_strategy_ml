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

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning'
mf_fac = pd.read_pickle(data_pat + '/mf/sharpe_weight_1/fac.pkl')  # 记得修改
mf_fac['反转因子改进_日频资金流'] = mf_fac.pop('反转因子改进')
hfmf_fac = pd.read_pickle(data_pat + '/hfmf/sharpe_weight_1/fac.pkl')  # 记得修改
hfmf_fac['反转因子改进_高频资金流'] = hfmf_fac.pop('反转因子改进')
fac_comb = dict(mf_fac, **hfmf_fac)

# 基础函数，计算因子之间的相关系数
def cal_factor_corr(fac_dict, pat_str):
    if not os.path.exists(pat_str):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(pat_str)
    total_data = pd.concat(fac_dict.values(), keys=fac_dict.keys())
    total_data = total_data.reset_index().set_index('level_1')
    corank_total = total_data.groupby(total_data.index).apply(lambda g: g.set_index('level_0').T.corr('spearman'))
    co_rank = corank_total.groupby(corank_total.index.get_level_values(1)).mean()
    co_rank = co_rank.reindex(co_rank.columns)  # 调整顺序，化为对称阵
    co_rank.to_csv(pat_str + "/mf_hfmf_cluster/mf_hfmf_rank_corr.csv", index=True, encoding='utf_8_sig')  # 记得修改
    return co_rank

# 新聚合因子之间的相关性
co_rank = cal_factor_corr(fac_comb, data_pat)
print(co_rank)

# 聚合方式（一）：同类别等权平均
new_fac = {}
comb = pd.concat([uc.cs_rank(fac_comb['高频资金流分布']), uc.cs_rank(fac_comb['日间资金流波动'])])
new_fac['资金流的稳定性'] = comb.groupby(comb.index).mean()
comb = pd.concat([uc.cs_rank(fac_comb['反转因子改进_日频资金流']), uc.cs_rank(fac_comb['反转因子改进_高频资金流'])])
new_fac['反转因子改进_资金流'] = comb.groupby(comb.index).mean()
comb = pd.concat([uc.cs_rank(fac_comb['高频资金流分布']), uc.cs_rank(fac_comb['日间资金流波动']), uc.cs_rank(fac_comb['主力流入流出占比'])])
new_fac['资金流的稳定性+主力流入流出占比'] = comb.groupby(comb.index).mean()
f = open(data_pat + '/mf_hfmf_cluster/eq/fac.pkl', 'wb')
pickle.dump(new_fac, f, -1)
f.close()

# 聚合方式（二）：同类别sharpe加权
new_fac = {}
comb = pd.concat([uc.cs_rank(fac_comb['高频资金流分布']) * 0.648901798, uc.cs_rank(fac_comb['日间资金流波动']) * 0.509416429])
new_fac['资金流的稳定性'] = comb.groupby(comb.index).mean()
comb = pd.concat([uc.cs_rank(fac_comb['反转因子改进_日频资金流']) * 0.06240904, uc.cs_rank(fac_comb['反转因子改进_高频资金流']) * 0.110874718])
new_fac['反转因子改进_资金流'] = comb.groupby(comb.index).mean()
comb = pd.concat([uc.cs_rank(fac_comb['高频资金流分布']) * 0.648901798, uc.cs_rank(fac_comb['日间资金流波动']) * 0.509416429,
                  uc.cs_rank(fac_comb['主力流入流出占比']) * 0.345566647])
new_fac['资金流的稳定性+主力流入流出占比'] = comb.groupby(comb.index).mean()
f = open(data_pat + '/mf_hfmf_cluster/sharpe_weight/fac.pkl', 'wb')
pickle.dump(new_fac, f, -1)
f.close()

# 聚合方式（三）：所有等权平均
new_fac = {}
comb = pd.concat([uc.cs_rank(fac_comb['高频资金流分布']), uc.cs_rank(fac_comb['日间资金流波动']),
                  uc.cs_rank(fac_comb['主力流入流出占比']), uc.cs_rank(fac_comb['反转因子改进_日频资金流']),
                  uc.cs_rank(fac_comb['反转因子改进_高频资金流']), uc.cs_rank(fac_comb['大单行为']),
                  uc.cs_rank(fac_comb['主力单数行为'])])
new_fac['资金流'] = comb.groupby(comb.index).mean()
f = open(data_pat + '/mf_hfmf_cluster/all_eq/fac.pkl', 'wb')
pickle.dump(new_fac, f, -1)
f.close()

# 聚合方式（四）：所有按sharpe比率加权平均
new_fac = {}
comb = pd.concat([uc.cs_rank(fac_comb['高频资金流分布']) * 0.648901798, uc.cs_rank(fac_comb['日间资金流波动']) * 0.509416429,
                  uc.cs_rank(fac_comb['主力流入流出占比']) * 0.345566647, uc.cs_rank(fac_comb['反转因子改进_日频资金流']) * 0.06240904,
                  uc.cs_rank(fac_comb['反转因子改进_高频资金流']) * 0.110874718, uc.cs_rank(fac_comb['大单行为']) * 0.318672397,
                  uc.cs_rank(fac_comb['主力单数行为']) * 0.355195913])
new_fac['资金流'] = comb.groupby(comb.index).mean()
f = open(data_pat + '/mf_hfmf_cluster/all_sharpe_weight/fac.pkl', 'wb')
pickle.dump(new_fac, f, -1)
f.close()

"""
# 把聚合因子的表现结果汇总
type = 'all_sharpe_weight'  # 记得修改
perf_path = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/mf_hfmf_cluster/' + str(type) + '/eq_tvwap'
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
