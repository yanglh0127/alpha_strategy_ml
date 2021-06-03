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
from matplotlib import pyplot as plt
import statsmodels.api as sm

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/5group/linear_model'  # 记得修改

begin = '2015-01-01'
end = '2021-03-31'  # 记得修改
trade_days = query_data.get_trade_days('d', from_trade_day=begin, to_trade_day=end)

new_f = pd.read_pickle(data_pat + '/new_f.pkl')
new_f = new_f.dropna(how='any')  # 所有因子值都不为空

# 每天的截面回归估计因子暴露
coef = {}
R_sq = {}
for date in trade_days:
    sub_data = new_f.loc[date,]
    model = sm.OLS(sub_data.iloc[:, -1], sm.add_constant(sub_data.iloc[:, 0:-1]), missing='drop').fit()  # 市值和行业变量?
    coef[date] = model.params
    R_sq[date] = model.rsquared_adj
    print(date)

coef_param = pd.concat(coef.values(), axis=1, keys=coef.keys()).T
coef_param.index = pd.to_datetime(coef_param.index)
r2_param = pd.DataFrame(R_sq.values(), index=R_sq.keys(), columns=['R_square_adj'])
plt.figure()
plt.plot(r2_param.index, r2_param['R_square_adj'])
plt.plot(r2_param.index, r2_param['R_square_adj'].rolling(20).mean())
plt.show()
coef_param.to_csv(data_pat + '/ols/coef_param.csv',encoding='gbk')
r2_param.to_csv(data_pat + '/ols/r2_param.csv',encoding='gbk')

# 求收益率预测值
new_f['const'] = 1
new_f = new_f.drop(['stock_rela'], axis=1)
fac = {}

# 只用最近一次截面回归得到的系数
coef_param2 = pd.concat([new_f.reset_index(level=1).iloc[:, 0], uc.ts_delay(coef_param, 11)], axis=1)  # 11天后才能用估计出的参数，记得修改
coef_param2 = coef_param2.set_index([coef_param2.index, 'level_1'])
pred = (coef_param2 * new_f).sum(axis=1, min_count=1)
pred = pred.unstack()
pred = pred.dropna(how='all')
fac['nearest'] = pred

# 用过去20日截面回归得到的系数的平均值
coef_param3 = pd.concat([new_f.reset_index(level=1).iloc[:, 0], uc.ts_delay(coef_param.rolling(20).mean(), 11)], axis=1)  # 11天后才能用估计出的参数，记得修改
coef_param3 = coef_param3.set_index([coef_param3.index, 'level_1'])
pred2 = (coef_param3 * new_f).sum(axis=1, min_count=1)
pred2 = pred2.unstack()
pred2 = pred2.dropna(how='all')
fac['ma_20'] = pred2

# 用过去60日截面回归得到的系数的平均值
coef_param4 = pd.concat([new_f.reset_index(level=1).iloc[:, 0], uc.ts_delay(coef_param.rolling(60).mean(), 11)], axis=1)  # 11天后才能用估计出的参数，记得修改
coef_param4 = coef_param4.set_index([coef_param4.index, 'level_1'])
pred3 = (coef_param4 * new_f).sum(axis=1, min_count=1)
pred3 = pred3.unstack()
pred3 = pred3.dropna(how='all')
fac['ma_60'] = pred3

# 用过去120日截面回归得到的系数的平均值
coef_param5 = pd.concat([new_f.reset_index(level=1).iloc[:, 0], uc.ts_delay(coef_param.rolling(120).mean(), 11)], axis=1)  # 11天后才能用估计出的参数，记得修改
coef_param5 = coef_param5.set_index([coef_param5.index, 'level_1'])
pred4 = (coef_param5 * new_f).sum(axis=1, min_count=1)
pred4 = pred4.unstack()
pred4 = pred4.dropna(how='all')
fac['ma_120'] = pred4

# 用过去240日截面回归得到的系数的平均值
coef_param6 = pd.concat([new_f.reset_index(level=1).iloc[:, 0], uc.ts_delay(coef_param.rolling(240).mean(), 11)], axis=1)  # 11天后才能用估计出的参数，记得修改
coef_param6 = coef_param6.set_index([coef_param6.index, 'level_1'])
pred5 = (coef_param6 * new_f).sum(axis=1, min_count=1)
pred5 = pred5.unstack()
pred5 = pred5.dropna(how='all')
fac['ma_240'] = pred5

# 用过去480日截面回归得到的系数的平均值
coef_param7 = pd.concat([new_f.reset_index(level=1).iloc[:, 0], uc.ts_delay(coef_param.rolling(480).mean(), 11)], axis=1)  # 11天后才能用估计出的参数，记得修改
coef_param7 = coef_param7.set_index([coef_param7.index, 'level_1'])
pred6 = (coef_param7 * new_f).sum(axis=1, min_count=1)
pred6 = pred6.unstack()
pred6 = pred6.dropna(how='all')
fac['ma_480'] = pred6

f = open(data_pat + '/ols/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac, f, -1)
f.close()
