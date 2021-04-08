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

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'
fac_meaning = pd.read_excel(data_pat + '/all.xlsx', sheet_name='各类聚合因子的表现', index_col=0)

fac_data = pd.read_pickle(data_pat + '/fac_last.pkl')
"""
# 聚合方式（一）：按逻辑相关性对同一类下所有因子进行等权聚合
fac_comb = {}
temp = {}
for tag in fac_data.keys():
    temp[tag] = uc.cs_rank(fac_data[tag])
comb = pd.concat(temp.values())
fac_comb['all_eq'] = comb.groupby(comb.index).mean()
fac_comb['all_eq'].index = pd.to_datetime(fac_comb['all_eq'].index)
f = open(data_pat + '/all_eq/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
"""
# 聚合方式（二）：先选出sharp排名前50%的因子，再进行等权聚合
fac_choose = fac_meaning[fac_meaning['sharp_ratio'].rank(pct=True) >= 0.5].index.tolist()
fac_comb = {}
temp = {}
for tag in fac_choose:
    temp[tag] = uc.cs_rank(fac_data[tag])
comb = pd.concat(temp.values())
fac_comb['50%_eq'] = comb.groupby(comb.index).mean()
fac_comb['50%_eq'].index = pd.to_datetime(fac_comb['50%_eq'].index)
f = open(data_pat + '/50%_eq/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
"""
# 聚合方式（三）：先选出sharp大于0的因子，再按sharpe比率进行加权聚合
fac_choose = fac_meaning[fac_meaning['sharp_ratio'] > 0][['sharp_ratio']]
fac_comb = {}
temp = {}
for tag in fac_choose.index.tolist():
    temp[tag] = uc.cs_rank(fac_data[tag]) * fac_choose.loc[tag, 'sharp_ratio']
comb = pd.concat(temp.values())
fac_comb['sharpe_weight'] = comb.groupby(comb.index).mean()
fac_comb['sharpe_weight'].index = pd.to_datetime(fac_comb['sharpe_weight'].index)
f = open(data_pat + '/sharpe_weight/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
"""
# 聚合方式（四）：先选出sharp排名前30%的因子，再进行等权聚合
fac_choose = fac_meaning[fac_meaning['sharp_ratio'].rank(pct=True) >= 0.7].index.tolist()
fac_comb = {}
temp = {}
for tag in fac_choose:
    temp[tag] = uc.cs_rank(fac_data[tag])
comb = pd.concat(temp.values())
fac_comb['30%_eq'] = comb.groupby(comb.index).mean()
fac_comb['30%_eq'].index = pd.to_datetime(fac_comb['30%_eq'].index)
f = open(data_pat + '/30%_eq/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
# 聚合方式（五）：先选出sharp排名前20%的因子，再进行等权聚合
fac_choose = fac_meaning[fac_meaning['sharp_ratio'].rank(pct=True) >= 0.8].index.tolist()
fac_comb = {}
temp = {}
for tag in fac_choose:
    temp[tag] = uc.cs_rank(fac_data[tag])
comb = pd.concat(temp.values())
fac_comb['20%_eq'] = comb.groupby(comb.index).mean()
fac_comb['20%_eq'].index = pd.to_datetime(fac_comb['20%_eq'].index)
f = open(data_pat + '/20%_eq/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()

"""
# 把聚合因子的表现结果汇总
type = '50%_eq'  # 记得修改
perf_path = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster/' + str(type) + '/eq_tvwap'
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
