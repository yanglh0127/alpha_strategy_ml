import pandas as pd
from ft_platform.factor_process import fetch
import os
import h5py
from ft_platform.utils import utils_calculation as uc
from utils_func import query_data
import pickle

begin = '2017-01-01'
end = '2021-02-28'

# 读取王峰因子表现汇总 # user = 'wangfeng:847,847,845','yyzhao:304,220,176','JiahuLi:191,189','Alex:195,194','cshu:97,'
mine_summary = query_data.get_alphafactors_info(user='wangfeng')
print(len(mine_summary))
# 提取因子名
fac_name = [i['factor_name'] for i in mine_summary]
print(len(fac_name))
# 提取因子ic的正负
ic = [i['perf'][list(i['perf'].keys())[0]]['ic-mean'] for i in mine_summary]
print(len(ic))
ic_sign = [uc.sign(i['perf'][list(i['perf'].keys())[0]]['ic-mean']) for i in mine_summary]
print(len(ic_sign))
# 提取因子值
factor_value = fetch.fetch_factor(begin, end, fields=fac_name, standard='clean1_alla', codes=None, df=False)
# 有些因子数据不足1009个,比较奇怪
factor_value = {k: v for k, v in factor_value.items() if len(v) == len(query_data.get_trade_days('d', from_trade_day=begin, to_trade_day=end))}
# 调整因子正负
factor_value_adj = {}
for summa in mine_summary:
    if summa['factor_name'] in list(factor_value.keys()):
        factor_value_adj[summa['factor_name']] = factor_value[summa['factor_name']] * \
                                                 uc.sign(summa['perf'][list(summa['perf'].keys())[0]]['ic-mean'])

# 拆解
new_name = list(factor_value_adj.keys())
factor_value_adj1 = {k: factor_value_adj[k] for k in new_name[0:200]}
factor_value_adj2 = {k: factor_value_adj[k] for k in new_name[200:400]}
factor_value_adj3 = {k: factor_value_adj[k] for k in new_name[400:600]}
factor_value_adj4 = {k: factor_value_adj[k] for k in new_name[600:845]}

# 保存因子值
f = open('E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_wang_20170101-20210228/all_fac_wang1_20170101-20210228.pkl', 'wb')
pickle.dump(factor_value_adj1, f, -1)
f.close()
f = open('E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_wang_20170101-20210228/all_fac_wang2_20170101-20210228.pkl', 'wb')
pickle.dump(factor_value_adj2, f, -1)
f.close()
f = open('E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_wang_20170101-20210228/all_fac_wang3_20170101-20210228.pkl', 'wb')
pickle.dump(factor_value_adj3, f, -1)
f.close()
f = open('E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_wang_20170101-20210228/all_fac_wang4_20170101-20210228.pkl', 'wb')
pickle.dump(factor_value_adj4, f, -1)
f.close()
