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
from sklearn import linear_model

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/5group/linear_model'  # 记得修改
el_alpha = 0.1  # 记得修改
el_l1ratio = 0.00045  # 记得修改

begin = '2015-01-01'
end = '2021-03-31'  # 记得修改
trade_days = query_data.get_trade_days('d', from_trade_day=begin, to_trade_day=end)

new_f = pd.read_pickle(data_pat + '/new_f.pkl')
new_f = new_f.dropna(how='any')  # 所有因子值都不为空

def pool_elnet_pred(ro_wind, pre_wind, e_alp, e_lr):
    prediction = {}
    coef = {}
    for i in np.arange((ro_wind + pre_wind), len(trade_days), 1):
        # 截取样本区间pool在一起计算回归系数
        date_roll = pd.to_datetime(trade_days[(i - ro_wind - pre_wind):(i - pre_wind)])
        sub_data = new_f.loc[date_roll, :]
        # model = sm.OLS(sub_data.iloc[:, -1], sm.add_constant(sub_data.iloc[:, 0:-1]), missing='drop').fit()
        model = linear_model.ElasticNet(alpha=e_alp, l1_ratio=e_lr)
        model.fit(sub_data.iloc[:, 0:-1], sub_data.iloc[:, -1])
        coef[trade_days[i]] = pd.Series(model.coef_, index=sub_data.iloc[:, 0:-1].columns)  # 保留lasso估计的参数
        # cons = model.intercept_
        # 当前的因子值
        test_data = new_f.loc[pd.to_datetime(trade_days[i]), :]  # 参数隔（pred_window+1）天后才能用
        test_data = test_data.drop(['stock_rela'], axis=1)
        # 求y的预测值
        prediction[trade_days[i]] = pd.Series(model.predict(test_data), index=test_data.index)
        print(trade_days[i])
    pred = pd.concat(prediction, axis=1).T
    pred.index = pd.to_datetime(pred.index)
    cof = pd.concat(coef, axis=1).T
    cof.index = pd.to_datetime(cof.index)
    feature = (abs(cof) > 0).sum(axis=1)
    return pred, cof, feature

pred_result = {}
coef_result = {}
feature_result = {}
# pred_result['pool_20_' + str(lasso_cons)], coef_result['pool_20_' + str(lasso_cons)], feature_result['pool_20_' + str(lasso_cons)] = pool_lasso_pred(20, 10, lasso_cons)
# pred_result['pool_60_' + str(lasso_cons)], coef_result['pool_60_' + str(lasso_cons)], feature_result['pool_60_' + str(lasso_cons)] = pool_lasso_pred(60, 10, lasso_cons)
# pred_result['pool_120_' + str(lasso_cons)], coef_result['pool_120_' + str(lasso_cons)], feature_result['pool_120_' + str(lasso_cons)] = pool_lasso_pred(120, 10, lasso_cons)
# pred_result['pool_240_' + str(lasso_cons)], coef_result['pool_240_' + str(lasso_cons)], feature_result['pool_240_' + str(lasso_cons)] = pool_lasso_pred(240, 10, lasso_cons)
pred_result['pool_480_' + str(el_alpha) + '_' + str(el_l1ratio)], coef_result['pool_480_' + str(el_alpha) + '_' + str(el_l1ratio)], feature_result['pool_480_' + str(el_alpha) + '_' + str(el_l1ratio)] = pool_elnet_pred(480, 10, el_alpha, el_l1ratio)

f = open(data_pat + '/elnet/fac_' + str(el_alpha) + '_' + str(el_l1ratio) + '.pkl', 'wb')  # 记得修改
pickle.dump(pred_result, f, -1)
f.close()
f = open(data_pat + '/elnet/coef_' + str(el_alpha) + '_' + str(el_l1ratio) + '.pkl', 'wb')  # 记得修改
pickle.dump(coef_result, f, -1)
f.close()
f = open(data_pat + '/elnet/feat_' + str(el_alpha) + '_' + str(el_l1ratio) + '.pkl', 'wb')  # 记得修改
pickle.dump(feature_result, f, -1)
f.close()
