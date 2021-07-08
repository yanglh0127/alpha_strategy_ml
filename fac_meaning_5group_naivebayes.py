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
from sklearn.naive_bayes import GaussianNB

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/5group/linear_model'  # 记得修改

begin = '2015-01-01'
end = '2021-03-31'  # 记得修改
trade_days = query_data.get_trade_days('d', from_trade_day=begin, to_trade_day=end)

new_f = pd.read_pickle(data_pat + '/new_f.pkl')
new_f = new_f.dropna(how='any')  # 所有因子值都不为空
new_f['ud'] = new_f['stock_rela'].apply(lambda g: 1 if g > 0 else 0)
new_f = new_f.drop(['stock_rela'], axis=1)

def pool_nbayes_pred(ro_wind, pre_wind):
    prediction = {}
    for i in np.arange((ro_wind + pre_wind), len(trade_days), 1):
        # 截取样本区间pool在一起计算回归系数
        date_roll = pd.to_datetime(trade_days[(i - ro_wind - pre_wind):(i - pre_wind)])
        sub_data = new_f.loc[date_roll, :]
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        bay = GaussianNB().fit(sub_data.iloc[:, 0:-1], sub_data.iloc[:, -1])
        print("correct rate: ", bay.score(sub_data.iloc[:, 0:-1], sub_data.iloc[:, -1]))

        test_data = new_f.loc[pd.to_datetime(trade_days[i]), :]  # 参数隔（pred_window+1）天后才能用
        test_data = test_data.drop(['ud'], axis=1)
        prediction[trade_days[i]] = pd.Series(bay.predict_proba(test_data.values)[:, 1], index=test_data.index)
        print(trade_days[i])
    pred = pd.concat(prediction, axis=1).T
    pred.index = pd.to_datetime(pred.index)
    return pred

pred_result = {}
pred_result['pool_480'] = pool_nbayes_pred(480, 10)

f = open(data_pat + '/bayes/fac.pkl', 'wb')  # 记得修改
pickle.dump(pred_result, f, -1)
f.close()
