from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
# 随机森林前的数据处理
train_df = train_df.dropna(how='any')
x = train_df.iloc[:, :-1]
y = train_df.iloc[:, -1]
# 随机森林
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
clf = RandomForestClassifier(n_estimators=20, max_depth=3, min_samples_split=50, min_samples_leaf=20).fit(x.values, y.values)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print(clf.score(x.values, y.values))

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
clf2 = DecisionTreeClassifier(max_depth=3, min_samples_split=100, min_samples_leaf=50).fit(x.values, y.values)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print(clf2.score(x.values, y.values))
