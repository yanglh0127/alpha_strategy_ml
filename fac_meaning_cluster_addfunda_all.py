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
from itertools import combinations

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'
fac_meaning = pd.read_excel(data_pat + '/fac_addfunda/all_addfunda.xlsx', sheet_name='各类聚合因子的表现', index_col=0)

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
fac_all = dict(fac_old, **fac_fundamental)
"""
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

cal_factor_corr(fac_all, data_pat + '/fac_addfunda')
"""
"""
# 基本面的3类聚合因子，遍历所有组合，进行等权聚合
fac_choose = list(fac_fundamental.keys())
comb = []
for i in range(len(fac_choose)):
    comb.extend(list(combinations(fac_choose, i+1)))
fac_comb = {}
for com in comb:
    temp = {}
    for ele in com:
        temp[ele] = uc.cs_rank(fac_fundamental[ele])
    comb = pd.concat(temp.values())
    fac_comb['iter_funda_' + str(com) + '_eq'] = comb.groupby(comb.index).mean()
    fac_comb['iter_funda_' + str(com) + '_eq'].index = pd.to_datetime(fac_comb['iter_funda_' + str(com) + '_eq'].index)
f = open(data_pat + '/fac_addfunda/iter_funda_eq/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
"""
# 基本面的3类聚合因子，遍历所有组合，进行等权聚合
fac_choose = list(fac_fundamental.keys())
comb = []
for i in range(len(fac_choose)):
    comb.extend(list(combinations(fac_choose, i+1)))
fac_comb = {}
for com in comb:
    temp = {}
    for ele in com:
        temp[ele] = uc.cs_rank(fac_fundamental[ele]) * fac_meaning.loc[ele, 'sharp_ratio']
    comb = pd.concat(temp.values())
    fac_comb['iter_funda_' + str(com) + '_sharpe_weight'] = comb.groupby(comb.index).mean()
    fac_comb['iter_funda_' + str(com) + '_sharpe_weight'].index = pd.to_datetime(fac_comb['iter_funda_' + str(com) + '_sharpe_weight'].index)
f = open(data_pat + '/fac_addfunda/iter_funda_sharpe_weight/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
"""
# 聚合方式（一）：从夏普比率最高的那个聚合因子开始，依次加入下一个夏普比率最高的聚合因子进行等权聚合，遍历
fac_meaning = fac_meaning.sort_values(by='sharp_ratio',axis=0,ascending=False)
fac_comb = {}
for i in range(len(fac_meaning)):
    tag_list = fac_meaning.index[0:(i+1)]
    temp = {}
    for tag in tag_list:
        temp[tag] = uc.cs_rank(fac_all[tag])
    comb = pd.concat(temp.values())
    fac_comb['best' + str(i+1) + '_eq'] = comb.groupby(comb.index).mean()
    fac_comb['best' + str(i+1) + '_eq'].index = pd.to_datetime(fac_comb['best' + str(i+1) + '_eq'].index)
f = open(data_pat + '/fac_addfunda/best_eq/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
"""
# 聚合方式（二）：从夏普比率最高的那个聚合因子开始，依次加入下一个夏普比率最高的聚合因子进行sharpe比率加权聚合，遍历所有sharpe比率大于0的聚合因子
fac_meaning = fac_meaning[fac_meaning['sharp_ratio'] > 0]
fac_meaning = fac_meaning.sort_values(by='sharp_ratio', axis=0, ascending=False)
fac_comb = {}
for i in range(len(fac_meaning)):
    tag_list = fac_meaning.index[0:(i+1)]
    temp = {}
    for tag in tag_list:
        temp[tag] = uc.cs_rank(fac_all[tag]) * fac_meaning.loc[tag, 'sharp_ratio']
    comb = pd.concat(temp.values())
    fac_comb['best' + str(i+1) + '_sharpe_weight'] = comb.groupby(comb.index).mean()
    fac_comb['best' + str(i+1) + '_sharpe_weight'].index = pd.to_datetime(fac_comb['best' + str(i+1) + '_sharpe_weight'].index)
f = open(data_pat + '/fac_addfunda/best_sharpe_weight/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
# 聚合方式（三）：取出夏普比率排名前十的聚合因子，遍历所有的组合方式（2**n种），进行等权聚合
fac_meaning = fac_meaning.sort_values(by='sharp_ratio', axis=0, ascending=False)
fac_choose = fac_meaning.index[0:10]
comb = []
for i in range(len(fac_choose)):
    comb.extend(list(combinations(fac_choose, i+1)))
fac_comb = {}
for com in comb:
    temp = {}
    comb_name = '('
    for ele in com:
        temp[ele] = uc.cs_rank(fac_all[ele])
        if ele.split('_')[-2] == 'fundamental':
            comb_name = comb_name + ele.split('_')[-1] + ','
        else:
            comb_name = comb_name + ele.split('_')[-2] + ','
    comb = pd.concat(temp.values())
    comb_name = comb_name + ')'
    print(comb_name)
    fac_comb['iter_' + comb_name + '_eq'] = comb.groupby(comb.index).mean()
    fac_comb['iter_' + comb_name + '_eq'].index = pd.to_datetime(fac_comb['iter_' + comb_name + '_eq'].index)
# 拆解
new_name = list(fac_comb.keys())
factor_1 = {}
factor_1 = {k: fac_comb[k] for k in new_name[0:200]}  # 记得修改
f = open(data_pat + '/fac_addfunda/iter10_eq/fac_1.pkl', 'wb')  # 记得修改
pickle.dump(factor_1, f, -1)
f.close()
