import pandas as pd
import pickle
from matplotlib import pyplot as plt
import statsmodels.api as sm
import numpy as np
import math
from ft_platform.utils import utils_calculation as uc
from utils_func import query_data


data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'
new_f = pd.read_pickle(data_pat + '/fac_reshape.pkl')
fac_choose = ['50%_eq_1_高频资金流分布_hfmf', 'sharpe_weight_反转因子相关_vp', 'sharpe_weight_1_日间资金流波动_mf',
              '50%_eq_收益率和波动率的相关性_vp', 'sharpe_weight_1_收盘行为异常_hfvp', 'best_1_日间成交量(额)的波动率_vp',
              'sharpe_weight_1_日内成交额分布的稳定性_hfvp', 'next_re']
new_f = new_f[fac_choose]


trade_days = query_data.get_trade_days('d', 'SSE', '2017-01-01', '2021-02-28')  # 记得修改
trade_days = [pd.to_datetime(i) for i in trade_days]
# 每天的截面回归估计因子暴露
coef = {}
R_sq = {}
for date in trade_days:
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
coef_param.to_csv(data_pat + '/linear_regress_7/coef_param.csv',encoding='gbk')
r2_param.to_csv(data_pat + '/linear_regress_7/r2_param.csv',encoding='gbk')


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


# 求收益率预测值(只用最近一次截面回归得到的系数)
fac = {}
new_f['const'] = 1
new_f = new_f.drop(['next_re'], axis=1)
coef_param2 = pd.concat([new_f.reset_index(level=1).iloc[:, 0], uc.ts_delay(coef_param, 2)], axis=1)  # 2天后才能用估计出的参数
coef_param2 = coef_param2.set_index([coef_param2.index, 'level_1'])
pred = (coef_param2 * new_f).sum(axis=1, min_count=2)  # 至少包含一个变量和一个const
pred = pred.unstack()
pred = pred.dropna(how='all')
fac['nearest'] = pred
f = open(data_pat + '/linear_regress_7/nearest/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac, f, -1)
f.close()


# 求收益率预测值(用过去20日截面回归得到的系数的平均值)
fac = {}
coef_param3 = pd.concat([new_f.reset_index(level=1).iloc[:, 0], uc.ts_delay(coef_param.rolling(20).mean(), 2)], axis=1)  # 2天后才能用估计出的参数
coef_param3 = coef_param3.set_index([coef_param3.index, 'level_1'])
pred2 = (coef_param3 * new_f).sum(axis=1, min_count=2)  # 至少包含一个变量和一个const
pred2 = pred2.unstack()
pred2 = pred2.dropna(how='all')
fac['ma_20'] = pred2
f = open(data_pat + '/linear_regress_7/ma_20/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac, f, -1)
f.close()


# 求收益率预测值(用过去60日截面回归得到的系数的平均值)
fac = {}
coef_param4 = pd.concat([new_f.reset_index(level=1).iloc[:, 0], uc.ts_delay(coef_param.rolling(60).mean(), 2)], axis=1)  # 2天后才能用估计出的参数
coef_param4 = coef_param4.set_index([coef_param4.index, 'level_1'])
pred3 = (coef_param4 * new_f).sum(axis=1, min_count=2)  # 至少包含一个变量和一个const
pred3 = pred3.unstack()
pred3 = pred3.dropna(how='all')
fac['ma_60'] = pred3
f = open(data_pat + '/linear_regress_7/ma_60/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac, f, -1)
f.close()


# 求收益率预测值(用过去120日截面回归得到的系数的平均值)
fac = {}
coef_param5 = pd.concat([new_f.reset_index(level=1).iloc[:, 0], uc.ts_delay(coef_param.rolling(120).mean(), 2)], axis=1)  # 2天后才能用估计出的参数
coef_param5 = coef_param5.set_index([coef_param5.index, 'level_1'])
pred4 = (coef_param5 * new_f).sum(axis=1, min_count=2)  # 至少包含一个变量和一个const
pred4 = pred4.unstack()
pred4 = pred4.dropna(how='all')
fac['ma_120'] = pred4
f = open(data_pat + '/linear_regress_7/ma_120/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac, f, -1)
f.close()


# 求收益率预测值(用过去240日截面回归得到的系数的平均值)
fac = {}
coef_param6 = pd.concat([new_f.reset_index(level=1).iloc[:, 0], uc.ts_delay(coef_param.rolling(240).mean(), 2)], axis=1)  # 2天后才能用估计出的参数
coef_param6 = coef_param6.set_index([coef_param6.index, 'level_1'])
pred5 = (coef_param6 * new_f).sum(axis=1, min_count=2)  # 至少包含一个变量和一个const
pred5 = pred5.unstack()
pred5 = pred5.dropna(how='all')
fac['ma_240'] = pred5
f = open(data_pat + '/linear_regress_7/ma_240/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac, f, -1)
f.close()
