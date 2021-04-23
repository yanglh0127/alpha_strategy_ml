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
fac_meaning = pd.read_excel(data_pat + '/all.xlsx', sheet_name='各类聚合因子的表现', index_col=0)

fac_data = pd.read_pickle(data_pat + '/fac_last.pkl')
fac_select = ['50%_eq_1_高频资金流分布_hfmf', 'sharpe_weight_反转因子相关_vp', '50%_eq_收益率和波动率的相关性_vp',
              'sharpe_weight_1_收盘行为异常_hfvp', 'best1_1_主力单数行为_mf', 'sharpe_weight_量价相关性_vp',
              '50%_eq_1_大单行为_hfmf', 'sharpe_weight_1_日内收益率的分布_hfvp', 'best1_1_流动性因子改进_hfvp',
              'best1_1_高频贝塔_hfvp', 'sharpe_weight_情绪因子_vp']
fac_data = {k: v for k, v in fac_data.items() if k in fac_select}
fac_meaning = fac_meaning.loc[fac_select]

"""
# 聚合方式（六）：从夏普比率最高的那个聚合因子开始，依次加入下一个夏普比率最高的聚合因子进行等权聚合，遍历
fac_meaning = fac_meaning.sort_values(by='sharp_ratio',axis=0,ascending=False)
fac_comb = {}
for i in range(len(fac_meaning)):
    tag_list = fac_meaning.index[0:(i+1)]
    temp = {}
    for tag in tag_list:
        temp[tag] = uc.cs_rank(fac_data[tag])
    comb = pd.concat(temp.values())
    fac_comb['best' + str(i+1) + '_eq'] = comb.groupby(comb.index).mean()
    fac_comb['best' + str(i+1) + '_eq'].index = pd.to_datetime(fac_comb['best' + str(i+1) + '_eq'].index)
f = open(data_pat + '/fac_select/best_eq/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
"""
# 聚合方式（七）：从夏普比率最高的那个聚合因子开始，依次加入下一个夏普比率最高的聚合因子进行sharpe比率加权聚合，遍历所有sharpe比率大于0的聚合因子
fac_meaning = fac_meaning[fac_meaning['sharp_ratio'] > 0]
fac_meaning = fac_meaning.sort_values(by='sharp_ratio', axis=0, ascending=False)
fac_comb = {}
for i in range(len(fac_meaning)):
    tag_list = fac_meaning.index[0:(i+1)]
    temp = {}
    for tag in tag_list:
        temp[tag] = uc.cs_rank(fac_data[tag]) * fac_meaning.loc[tag, 'sharp_ratio']
    comb = pd.concat(temp.values())
    fac_comb['best' + str(i+1) + '_sharpe_weight'] = comb.groupby(comb.index).mean()
    fac_comb['best' + str(i+1) + '_sharpe_weight'].index = pd.to_datetime(fac_comb['best' + str(i+1) + '_sharpe_weight'].index)
f = open(data_pat + '/fac_select/best_sharpe_weight/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
# """
# 聚合方式（八）：遍历所有的组合方式（2**n种），进行等权聚合
fac_meaning = fac_meaning.sort_values(by='sharp_ratio', axis=0, ascending=False)
fac_choose = fac_meaning.index
comb = []
for i in range(len(fac_choose)):
    comb.extend(list(combinations(fac_choose, i+1)))
fac_comb = {}
for com in comb:
    temp = {}
    comb_name = '('
    for ele in com:
        temp[ele] = uc.cs_rank(fac_data[ele])
        comb_name = comb_name + ele.split('_')[-2] + ','
    comb = pd.concat(temp.values())
    comb_name = comb_name + ')'
    print(comb_name)
    fac_comb['iter_' + comb_name + '_eq'] = comb.groupby(comb.index).mean()
    fac_comb['iter_' + comb_name + '_eq'].index = pd.to_datetime(fac_comb['iter_' + comb_name + '_eq'].index)
# f = open(data_pat + '/iter_eq/fac.pkl', 'wb')  # 记得修改
# pickle.dump(fac_comb, f, -1)
# f.close()
# """
"""
# 聚合方式（九）：从sharpe比率排名前七的聚合因子里，遍历所有的组合方式（2**n种），进行sharp比率加权聚合
fac_meaning = fac_meaning.sort_values(by='sharp_ratio', axis=0, ascending=False)
fac_choose = fac_meaning.index[0:7]
comb = []
for i in range(len(fac_choose)):
    comb.extend(list(combinations(fac_choose, i+1)))
fac_comb = {}
for com in comb:
    temp = {}
    for ele in com:
        temp[ele] = uc.cs_rank(fac_data[ele]) * fac_meaning.loc[ele, 'sharp_ratio']
    comb = pd.concat(temp.values())
    fac_comb['iter7_' + str(com) + '_sharpe_weight'] = comb.groupby(comb.index).mean()
    fac_comb['iter7_' + str(com) + '_sharpe_weight'].index = pd.to_datetime(fac_comb['iter7_' + str(com) + '_sharpe_weight'].index)
f = open(data_pat + '/iter7_sharpe_weight/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
