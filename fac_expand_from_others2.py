import pandas as pd
import os
import json
from utils_func import query_data
from ft_platform.factor_process import fetch
import time
from ft_platform.utils import utils_calculation as uc
import pickle

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'

mine_data = pd.read_pickle('E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_20170101-20210228/all_fac_20170101-20210228.pkl')
other_data1 = pd.read_pickle(data_pat + '/fac_expand/other_facadj_1_20170101-20210228.pkl')
other_data2 = pd.read_pickle(data_pat + '/fac_expand/other_facadj_2_20170101-20210228.pkl')
other_data3 = pd.read_pickle(data_pat + '/fac_expand/other_facadj_3_20170101-20210228.pkl')
other_data4 = pd.read_pickle(data_pat + '/fac_expand/other_facadj_4_20170101-20210228.pkl')
other_data5 = pd.read_pickle(data_pat + '/fac_expand/other_facadj_5_20170101-20210228.pkl')
other_data6 = pd.read_pickle(data_pat + '/fac_expand/other_facadj_6_20170101-20210228.pkl')
all_data = dict(mine_data, **other_data1, **other_data2, **other_data3, **other_data4, **other_data5, **other_data6)

with open(data_pat + "/fac_structure.json",'r') as f:
    fac_structure = json.load(f)
with open(data_pat + "/other_fac_structure.json",'r') as f:
    other_fac_structure = json.load(f)

# 合并因子结构
all_fac = {'vp': {}, 'hfvp': {}}
for type, v in fac_structure.items():
    for tag, fac_names in v.items():
        if type in ['vp', 'hfvp']:
            all_fac[type][tag] = fac_names + other_fac_structure[type][tag]
            print(type, tag, len(all_fac[type][tag]))


# 因子聚合方式（一）：同一类别下等权组合
fac_comb = {}
for type, v in all_fac.items():
    for tag, fac_names in v.items():
        print(type, tag, len(fac_names))
        temp = {}
        for fac_name in fac_names:
            if fac_name not in ['factor_20216_vp', 'factor_90007_daily_vp']:  # 这两个因子似乎略有问题
                temp[fac_name] = uc.cs_rank(all_data[fac_name])
        print('concat')
        comb = pd.concat(temp.values())
        print('mean')
        fac_comb['all_eq_1_' + tag + '_' + type] = comb.groupby(comb.index).mean()
        fac_comb['all_eq_1_' + tag + '_' + type].index = pd.to_datetime(fac_comb['all_eq_1_' + tag + '_' + type].index)