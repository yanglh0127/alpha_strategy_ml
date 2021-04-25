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
coef_param = uc.ts_delay(coef_param, 2)  # 2天后才能用估计出的参数
r2_param = uc.ts_delay(r2_param, 2)  # 2天后才能用估计出的参数
coef_param = coef_param.groupby(coef_param.index.strftime('%Y-%m')).mean()  # 每个月更新一次权重
r2_param = r2_param.groupby(r2_param.index.strftime('%Y-%m')).mean()  # 每个月更新一次权重
plt.figure()
plt.plot(r2_param.index, r2_param['R_square_adj'])
plt.show()
coef_param.to_csv(data_pat + '/linear_regress_7_m/coef_param.csv',encoding='gbk')
r2_param.to_csv(data_pat + '/linear_regress_7_m/r2_param.csv',encoding='gbk')


# 画出因子暴露时间序列
le = np.size(coef_param, 0)
la = math.ceil(4*(le/100)**(2/9))
for coef_name in coef_param.columns:
    plt.figure()
    plt.plot(coef_param.index, coef_param[coef_name])
    plt.title(coef_name, fontproperties="SimSun")
    plt.show()
    model = sm.OLS(coef_param[coef_name], [1 for i in range(le)]).fit(cov_type='HAC', cov_kwds={'maxlags': la})
    print(model.summary())  # 有些因子的系数显著为负?多因子回归的影响


# 求收益率预测值
coef_param = pd.DataFrame({i: uc.ts_delay(coef_param, 1).loc[i.strftime('%Y-%m')] for i in trade_days}).T  # 整个月都用上个月算出的权重
fac = {}
coef_param3 = pd.concat([new_f.reset_index(level=1).iloc[:, 0], coef_param], axis=1)
coef_param3 = coef_param3.set_index([coef_param3.index, 'level_1'])
pred2 = (coef_param3 * new_f).sum(axis=1, min_count=2)  # 至少包含一个变量和一个const
pred2 = pred2.unstack()
pred2 = pred2.dropna(how='all')
fac['1m_1m'] = pred2
f = open(data_pat + '/linear_regress_7_m/1m_1m/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac, f, -1)
f.close()
