import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from ft_platform.utils import utils_calculation as uc
from utils_func import query_data
from ft_platform import fetch_data
from ft_platform.factor_process import fetch
import time
import pickle

roll_window = 240
pred_window = 1
begin = '2017-01-01'
end = '2020-08-31'

name_pat = 'E:/Share/FengWang/Alpha/mine/'
hfmf_name = pd.read_pickle(name_pat + 'hfmf_factor/cluster_name.pkl')
hfvp_name = pd.read_pickle(name_pat + 'hfvp_factor/cluster_name.pkl')
mf_name = pd.read_pickle(name_pat + 'mf_factor/cluster_name.pkl')
vp_name = pd.read_pickle(name_pat + 'price_volume/cluster_name.pkl')
# factor_name = dict(hfmf_name, **hfvp_name, **mf_name, **vp_name)  # 记得修改
factor_name = hfvp_name  # 记得修改
name_list = []
for na in factor_name.values():
    name_list.extend(na)

# 读取因子数据
factor_value = fetch.fetch_factor(begin, end, fields=name_list, standard='clean1_alla', codes=None, df=False)
mine_summary = query_data.get_alphafactors_info(user='LihaiYang')
# 调整正负
factor_value_adj = {}
for summa in mine_summary:
    if summa['factor_name'] in list(factor_value.keys()):
        if 'IC' in list(summa['perf']['1_d'].keys()):
            factor_value_adj[summa['factor_name']] = factor_value[summa['factor_name']] * uc.sign(summa['perf']['1_d']['IC'])
        else:
            factor_value_adj[summa['factor_name']] = factor_value[summa['factor_name']] * uc.sign(summa['perf']['1_d']['ic-mean'])

# 建立股票在未来n日的涨跌标签
oc_data = fetch_data.fetch(begin, end, ['stock_adjopen', 'stock_adjclose'])
ud_tag = uc.ts_delay(oc_data['stock_adjclose'], -pred_window) / uc.ts_delay(oc_data['stock_adjopen'], -1) - 1  # 以第二日的开盘价买入
ud_tag = ud_tag.mask(ud_tag > 0, 1)
ud_tag = ud_tag.mask(ud_tag < 0, 0)

# 股票因子值的reshape
new_f = {}
for k, v in factor_value_adj.items():
    new_v = pd.DataFrame(v.stack())
    new_v.columns = [k]
    new_f[k] = new_v
new_f = pd.concat(new_f.values(), axis=1)

# 滚动生成上涨概率预测
prediction = {}
bay = GaussianNB()
for i in np.arange((roll_window + pred_window), len(ud_tag) + 1, 1):
    ud_tag_temp = ud_tag.iloc[(i - roll_window - pred_window):(i - pred_window), :]
    a = pd.DataFrame(ud_tag_temp.stack())
    a.columns = ['ud']
    train_df = pd.concat([new_f, a], join='inner', axis=1)
    test_df = new_f.loc[ud_tag.index[i - 1]]
    # 贝叶斯分类前的数据处理
    train_df = train_df.dropna(how='any')
    x = train_df.iloc[:, :-1]
    y = train_df.iloc[:, -1]
    # 贝叶斯分类
    bay.fit(x.values, y.values)
    print("correct rate: ", bay.score(x.values, y.values))
    pred = bay.predict_proba(test_df.values)[:, 1]  # 未来n日涨的概率的预测值
    prediction[ud_tag.index[i - 1]] = pd.DataFrame(pred[np.newaxis, :], index=[ud_tag.index[i - 1]], columns=test_df.index)
    print(ud_tag.index[i - 1])
prediction = pd.concat(prediction.values())
pred_result = {}
pred_result['240_1'] = prediction
f = open('E:/FT_Users/LihaiYang/Files/factor_comb_data/ml_comb/naive_bayes/240_1.pkl', 'wb')  # 记得修改
pickle.dump(pred_result, f, -1)
f.close()
