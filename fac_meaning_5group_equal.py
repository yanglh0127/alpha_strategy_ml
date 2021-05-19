from ft_platform import fetch_data
from ft_platform.utils import utils_calculation as uc
import pandas as pd
import pickle
from utils_func import query_data
from ft_platform.factor_process import fetch
from copy import deepcopy
import numpy as np
import time
import json

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/5group'  # 记得修改

# 计算未来1、3、5、10、20日收益率，以开盘1小时tvwap为标准
begin = '2015-01-01'  # 记得修改
end1 = '2021-03-31'
trade_days = query_data.get_trade_days('d', from_trade_day=begin, to_trade_day=end1)

# 读取最后选取的因子文件和对应的权重
with open(data_pat + "/fac_chosen.json",'r') as f:
    fac_choose = json.load(f)

# 因子分组情况和所有的因子list
factor_group = {}
factor_list = []
for k, v in fac_choose.items():
    temp = []
    for m, n in v.items():
        temp.extend(list(n.keys()))
    factor_group[k] = list(set(temp))
    factor_list.extend(list(set(temp)))

# 查看同样的因子在不同的样本和周期下权重是否有不同
temp = {}
for k, v in fac_choose.items():
    for m, n in v.items():
        for p, q in n.items():
            if p in list(temp.keys()):
                temp[p].append(q)
            else:
                temp[p] = [q]

for k, v in temp.items():
    temp[k] = list(set(v))

# 生成因子对应的权重
fac_weight = {}
for k, v in factor_group.items():
    tp = {}
    for fac in v:
        tp[fac] = temp[fac][0]
    fac_weight[k] = tp

"""
# 读取因子数据  # price_reversal: 59; volume_std: 35, 30; vp_corr: 9; financial report: 6, 5; financial forword: 18, 13
print('fetch')
fac_data = fetch.fetch_factor(begin, end1, fields=list(set(factor_list)), standard='clean1_alla', codes=None, df=False)
f = open(data_pat + '/all_eq/fac.pkl', 'wb')
pickle.dump(fac_data, f, -1)
f.close()
"""
fac_data = pd.read_pickle(data_pat + '/all_eq/fac.pkl')

# 检测因子  126,126,115  # 特别是高频数据，从15年才有数据，有些因子要用到几个月前的数据?
fac_data = {k: v for k, v in fac_data.items() if len(v) == len(trade_days)}  # 去掉数据缺失的因子

# top2000股票池
cap_data = fetch_data.fetch(begin, end1, ['stock_tcap'])
cap_rank = cap_data['stock_tcap'].rank(axis=1, ascending=False)
# 每日的top2000股票标记为1，否则为nan
top2000 = (cap_rank <= 2000).where((cap_rank <= 2000) == 1)  # 2015年8月6日只有1999只?

# 根据top2000股票池把因子值在非2000的置为空值
fac_data = {k: (v * top2000) for k, v in fac_data.items()}

# 所有的因子等权聚合
fac_comb = pd.DataFrame()
for k, v in temp.items():
    if k in fac_data.keys():
        if len(fac_comb) == 0:
            fac_comb = (fac_data[k] * v[0]).rank(axis=1)
        else:
            fac_comb = fac_comb + (fac_data[k] * v[0]).rank(axis=1)
a = fac_comb.notna().sum(axis=1)
print(a.min())
print(a.max())
fac_comb = {'fac_choose_comb': fac_comb}
f = open(data_pat + '/all_eq/fac_comb.pkl', 'wb')
pickle.dump(fac_comb, f, -1)
f.close()

# 所有的因子在组内等权组合，所有组再等权聚合
fac_comb1 = {}
for k, v in fac_weight.items():
    tp = pd.DataFrame()
    for p, q in v.items():
        if p in fac_data.keys():
            if len(tp) == 0:
                tp = (fac_data[p] * q).rank(axis=1)
            else:
                tp = tp + (fac_data[p] * q).rank(axis=1)
    fac_comb1[k] = tp
    a = tp.notna().sum(axis=1)
    print(a.min())
    print(a.max())

tp = pd.DataFrame()
for k, v in fac_comb1.items():
    if len(tp) == 0:
        tp = v.rank(axis=1)
    else:
        tp = tp + v.rank(axis=1)
fac_comb1['fac_comb'] = tp
a = tp.notna().sum(axis=1)
print(a.min())
print(a.max())
f = open(data_pat + '/group_eq/fac_comb.pkl', 'wb')
pickle.dump(fac_comb1, f, -1)
f.close()
