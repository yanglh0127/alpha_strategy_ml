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

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/5group/linear_model'  # 记得修改

begin = '2015-01-01'
end = '2021-03-31'  # 记得修改
trade_days = query_data.get_trade_days('d', from_trade_day=begin, to_trade_day=end)

new_f = pd.read_pickle(data_pat + '/new_f.pkl')
new_f = new_f.dropna(how='any')  # 所有因子值都不为空

coef = pd.read_pickle(data_pat + '/pls/coef_6.pkl')
coef = coef[list(coef.keys())[0]]

def pool_pred(ro_wind, pre_wind):
    prediction = {}
    update_time = np.arange((ro_wind + pre_wind), len(trade_days), 20)  # 隔20天更新一下权重
    for i in np.arange((ro_wind + pre_wind), len(trade_days), 1):
        # 当前的因子值
        test_data = new_f.loc[pd.to_datetime(trade_days[i]), :]
        test_data = test_data.drop(['stock_rela'], axis=1)
        if i in update_time:
            weig = coef.loc[pd.to_datetime(trade_days[i]), :]
        # 求y的预测值
        prediction[trade_days[i]] = (test_data * weig).sum(axis=1)
        print(trade_days[i])
    pred = pd.concat(prediction, axis=1).T
    pred.index = pd.to_datetime(pred.index)
    return pred

pred_result = {}
pred_result['pool_480'] = pool_pred(480, 10)

f = open(data_pat + '/pls_m/fac.pkl', 'wb')  # 记得修改
pickle.dump(pred_result, f, -1)
f.close()
