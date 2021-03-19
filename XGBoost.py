"""
import pandas as pd
from ft_platform.utils import utils_calculation as uc
from utils_func import query_data
from ft_platform import fetch_data
from ft_platform.factor_process import fetch
import time
import pickle
import numpy as np

roll_window = 240
pred_window = 1
begin = '2017-01-01'
end = '2017-12-31'

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

a = pd.DataFrame(ud_tag.stack())
a.columns = ['ud']
train_df = pd.concat([new_f, a], join='inner', axis=1)
# XGBoost前的数据处理
train_df = train_df.dropna(how='any')
x = train_df.iloc[:, :-1]
y = train_df.iloc[:, -1]
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
clf =
print(clf.score(x.values, y.values))
print(clf.feature_importances_[clf.feature_importances_ > 0].shape)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance

digits = datasets.load_digits()

### data analysis
print(digits.data.shape)
print(digits.target.shape)

### 划分训练集测试集
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=33)
### 训练模型
model = XGBClassifier(learning_rate=0.1,
                      n_estimators=100,  # 树的个数--100棵树建立xgboost
                      max_depth=6,  # 树的深度
                      min_child_weight=1,  # 叶子节点最小权重
                      gamma=0.,  # 惩罚项中叶子结点个数前的参数
                      subsample=0.8,  # 随机选择80%样本建立决策树
                      colsample_btree=0.8,  # 随机选择80%特征建立决策树
                      objective='multi:softmax',  # 指定损失函数
                      scale_pos_weight=1,  # 解决样本个数不平衡的问题
                      random_state=27  # 随机数
                      )
# 拟合
model.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric="mlogloss", early_stopping_rounds=10,
          verbose=True)

### 特征重要性
fig, ax = plt.subplots(figsize=(15, 15))
plot_importance(model, height=0.5, ax=ax, max_num_features=64)
plt.show()

### 预测
y_pred = model.predict(x_test)

### 模型正确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率: %.2f%%" % (accuracy * 100.0))
