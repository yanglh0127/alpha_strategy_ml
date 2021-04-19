import pandas as pd
import pickle
import numpy as np
import math
from ft_platform.utils import utils_calculation as uc
from utils_func import query_data
import time
from sklearn.ensemble import RandomForestClassifier

win = 240  # 选用估计随机森林参数的历史窗口

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'
new_f = pd.read_pickle(data_pat + '/fac_reshape.pkl')
new_f = new_f.dropna(how='any')
new_f['next_re_istop'] = (new_f['next_re'].groupby(new_f.index.get_level_values(0)).rank(pct=True) >= 0.9).astype(int)  # 收益率处于最高的前10%标志
new_f = new_f.drop(['next_re'], axis=1)

trade_days = query_data.get_trade_days('d', 'SSE', '2017-01-01', '2021-02-28')  # 记得修改
trade_days = [pd.to_datetime(i) for i in trade_days]
prediction = {}
for i in range(len(trade_days) - 1):
    if i >= win:
        train_df = new_f.loc[trade_days[(i-win):i]]
        x = train_df.iloc[:, :-1]
        y = train_df.iloc[:, -1]
        test_df = new_f.loc[trade_days[(i+1)]].iloc[:, :-1]
        # 随机森林
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        clf = RandomForestClassifier(n_estimators=20, max_depth=3, min_samples_split=50, min_samples_leaf=20).fit(x.values, y.values)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("correct rate: ", clf.score(x.values, y.values))
        pred = clf.predict_proba(test_df.values)[:, 1]  # 未来n日涨的概率的预测值
        prediction[trade_days[(i+1)]] = pd.DataFrame(pred[np.newaxis, :], index=[trade_days[(i+1)]], columns=test_df.index)
        print(trade_days[(i+1)])
prediction = pd.concat(prediction.values())
fac = {}
fac['window_' + str(win)] = prediction
f = open(data_pat + '/random_forest/window_' + str(win) + '/fac.pkl', 'wb')  # 记得修改
pickle.dump(fac, f, -1)
f.close()
