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
import time
from ft_platform import fetch_data
import statsmodels.api as sm
import numpy as np
import math


data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'
fac_meaning = pd.read_excel(data_pat + '/all.xlsx', sheet_name='各类聚合因子的表现', index_col=0)

fac_data = pd.read_pickle(data_pat + '/fac_last.pkl')  # 用本身还是用uc.cs_rank()?
fac_data = {k: uc.cs_standardize(v) for k, v in fac_data.items()}  # 是否先截面标准化?
# """
# 因子描述性统计
factor_describe = {}
for fac in fac_data.keys():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # 保存因子的描述性统计
    factor_describe[fac] = fac_data[fac].T.describe().T
    # 打印因子的描述性统计均值
    print(fac, fac_data[fac].T.describe().mean(axis=1))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# """
# 以下一日的开盘1小时tvwap到再下一日的开盘1小时tvwap收益率作为预测目标
begin = '2017-01-01'
end = '2021-03-02'
tvwap = fetch_data.fetch(begin, end, ['stock_twap_0930_1030'])  # adjtvwap?
fac_data['next_re'] = uc.ts_delay(tvwap['stock_twap_0930_1030'], -2) / uc.ts_delay(tvwap['stock_twap_0930_1030'], -1) - 1
fac_data['next_re'] = fac_data['next_re'].dropna(how='all')

# 将每天的对应数据合并
new_f = {}
for k, v in fac_data.items():
    new_v = pd.DataFrame(v.stack())
    new_v.columns = [k]
    new_f[k] = new_v
new_f = pd.concat(new_f.values(), axis=1)

# 每天的截面回归估计因子暴露
coef = {}
R_sq = {}
for date in fac_data['next_re'].index:
    sub_data = new_f.loc[date,]
    model = sm.OLS(sub_data.iloc[:, -1], sm.add_constant(sub_data.iloc[:, 0:-1]), missing='drop').fit()
    coef[date] = model.params
    R_sq[date] = model.rsquared_adj
    print(date)
coef_param = pd.concat(coef.values(), axis=1, keys=coef.keys())
coef_param = pd.DataFrame(coef_param.values.T, index=coef_param.columns, columns=coef_param.index)  # 转置
r2_param = pd.DataFrame(R_sq.values(), index=R_sq.keys(), columns=['R_square_adj'])
plt.figure()
plt.plot(r2_param.index, r2_param['R_square_adj'])
plt.plot(r2_param.index, r2_param['R_square_adj'].rolling(20).mean())
plt.show()

# 画出因子暴露时间序列
le = np.size(coef_param, 0)
la = math.ceil(4*(le/100)**(2/9))
for coef_name in coef_param.columns:
    plt.figure()
    plt.plot(coef_param.index, coef_param[coef_name])
    plt.plot(coef_param.index, coef_param[coef_name].rolling(20).mean())
    plt.title(coef_name, fontproperties="SimSun")
    plt.show()
    model = sm.OLS(coef_param[coef_name], [1 for i in range(le)]).fit(cov_type='HAC', cov_kwds={'maxlags': la})
    print(model.summary())  # 有些因子的系数显著为负?多因子回归的影响

# 求收益率预测值
new_f['const'] = 1
uc.ts_delay(coef_param, 2) * new_f  # 2天后才能用估计出的参数
