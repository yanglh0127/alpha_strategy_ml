import pandas as pd
from ft_platform.factor_process import fetch
import os
import h5py
from ft_platform.utils import utils_calculation as uc
from utils_func import query_data
import pickle

begin = '2020-09-01'
end = '2021-02-28'

name_pat = 'E:/Share/FengWang/Alpha/mine/'
hfmf_name = pd.read_pickle(name_pat + 'hfmf_factor/cluster_name.pkl')
hfvp_name = pd.read_pickle(name_pat + 'hfvp_factor/cluster_name.pkl')
mf_name = pd.read_pickle(name_pat + 'mf_factor/cluster_name.pkl')
vp_name = pd.read_pickle(name_pat + 'price_volume/cluster_name.pkl')
factor_name = dict(hfmf_name, **hfvp_name, **mf_name, **vp_name)
name_list = []
for na in factor_name.values():
    name_list.extend(na)

# 读取因子的样本外数据
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

hfmf_value = {}
path = 'E:/Share/FengWang/Alpha/mine/hfmf_factor/oos/clean'
for j in os.listdir(path):
    temp = h5py.File(path + '/' + j, 'r')
    hfmf_value[j[:-3]] = pd.DataFrame(temp['data'][:].astype(float),
                                      columns=temp['code'][:].astype(str),
                                      index=temp['trade_date'][:].astype(str))
all_fac = dict(factor_value_adj, **hfmf_value)

# 各类因子数据的样本外组合
print('cluster_comb')
fac_cluster = {}
for fac_gr, fac_names in factor_name.items():
    comb_temp = {}
    for fac_name in fac_names:
        comb_temp[fac_name] = uc.cs_rank(all_fac[fac_name])
    comb = pd.concat(comb_temp.values())
    fac_cluster[fac_gr] = comb.groupby(comb.index).mean()
    fac_cluster[fac_gr].index = pd.to_datetime(fac_cluster[fac_gr].index)

f = open('E:/FT_Users/LihaiYang/Files/factor_comb_data/all_cluster_comb_oos/simple_avg/9.pkl', 'wb')  # 这边路径记得改
pickle.dump(fac_cluster, f, -1)
f.close()

# 类与类之间因子数据的样本外组合
print('all_cluster')
cluster_num = 1  # 记得修改
all_cluster = {}
all_cluster_name = pd.read_pickle('E:/Share/FengWang/Alpha/mine/all_cluster_comb/cluster_' + str(cluster_num) + '/cluster_name.pkl')
for fac_gr, fac_names in all_cluster_name.items():
    comb_temp = {}
    for fac_name in fac_names:
        comb_temp[fac_name] = uc.cs_rank(fac_cluster[fac_name])
    comb = pd.concat(comb_temp.values())
    all_cluster[fac_gr] = comb.groupby(comb.index).mean()
    all_cluster[fac_gr].index = pd.to_datetime(all_cluster[fac_gr].index)
f = open('E:/FT_Users/LihaiYang/Files/factor_comb_data/all_cluster_comb_oos/simple_avg/' + str(cluster_num) + '.pkl', 'wb')  # 这边路径记得改
pickle.dump(all_cluster, f, -1)
f.close()
