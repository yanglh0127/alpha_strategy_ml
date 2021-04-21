import pandas as pd
import os
import json
from utils_func import query_data
from ft_platform.factor_process import fetch
import time
from ft_platform.utils import utils_calculation as uc
import pickle

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'

with open(data_pat + "/fac_structure.json",'r') as f:
    fac_structure = json.load(f)

mine_summary = query_data.get_alphafactors_info(user='LihaiYang')
mine_fac = [a['factor_name'] for a in mine_summary]

# 找出与自己因子相关性较高的其它人的因子
corr_choose = ['[0.9, 1.0)', '[0.5, 0.9)', '[-0.9, -0.5)', '[-1.0, -0.9)']  # 相关性大小选择
other_fac = {'hfmf': {}, 'mf': {}, 'vp': {}, 'hfvp': {}}
other_fac_all = []
for type, v in fac_structure.items():
    for tag, fac_names in v.items():
        other_fac[type][tag] = []
        for fac_name in fac_names:
            if type not in ['hfmf', 'mf']:  # 高频资金流因子没有alpha_information?暂时不做资金流向类的
                summ = query_data.get_alphafactors_info(factor_name=fac_name)
                stats_corr = summ[0]['stats_corr']['TOP2000_2017-06-30_2020-06-30']  # 相关系数的时间区间?
                for co, oth in stats_corr.items():
                    if co in corr_choose:
                        other_fac[type][tag].extend([fa for fa in oth[1] if fa not in mine_fac])  # 只纳入除自己因子外的其它人的因子
                        other_fac_all.extend([fa for fa in oth[1] if fa not in mine_fac])
        other_fac[type][tag] = list(set(other_fac[type][tag]))

with open(data_pat + "/other_fac_structure.json", "w") as f:
    json.dump(other_fac, f)
other_fac_all = list(set(other_fac_all))
with open(data_pat + "/other_fac_hcorr.json", "w") as f:
    json.dump(other_fac_all, f)


# 提取与自己因子相关性较高的其它人的因子值
print('fetch')
begin = '2017-01-01'
end = '2021-02-28'
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
factor_value = fetch.fetch_factor(begin, end, fields=other_fac_all, standard='clean1_alla', codes=None, df=False)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# 删除因子数据不足的因子
factor_value = {k: v for k, v in factor_value.items() if len(v) == len(query_data.get_trade_days('d', from_trade_day=begin, to_trade_day=end))}
f = open(data_pat + '/fac_expand/other_fac_20170101-20210228.pkl', 'wb')
pickle.dump(factor_value, f, -1)
f.close()

# 调整因子的正负
factor_value_adj = {}
for fac in factor_value.keys():
    fac_info = query_data.get_alphafactors_info(factor_name=fac)[0]
    if fac_info['user'] in ['wangfeng', 'JiahuLi', 'Alex']:
        factor_value_adj[fac] = factor_value[fac] * uc.sign(fac_info['perf'][list(fac_info['perf'].keys())[0]]['ic-mean'])
    if fac_info['user'] == 'yyzhao':
        factor_value_adj[fac] = factor_value[fac] * uc.sign(fac_info['perf'][list(fac_info['perf'].keys())[0]]['IC'])
    if fac_info['user'] == 'cshu':  # 舒畅的因子正负调整?
        factor_value_adj[fac] = factor_value[fac] * fac_info['perf_summary']['2018-03-09__2021-03-08']['1_d']['sign']

# 拆解
new_name = list(factor_value_adj.keys())
factor_value_adj1 = {k: factor_value_adj[k] for k in new_name[0:100]}
factor_value_adj2 = {k: factor_value_adj[k] for k in new_name[100:200]}
factor_value_adj3 = {k: factor_value_adj[k] for k in new_name[200:300]}
factor_value_adj4 = {k: factor_value_adj[k] for k in new_name[300:400]}
factor_value_adj5 = {k: factor_value_adj[k] for k in new_name[400:500]}
factor_value_adj6 = {k: factor_value_adj[k] for k in new_name[500:537]}

f = open(data_pat + '/fac_expand/other_facadj_1_20170101-20210228.pkl', 'wb')
pickle.dump(factor_value_adj1, f, -1)
f.close()
f = open(data_pat + '/fac_expand/other_facadj_2_20170101-20210228.pkl', 'wb')
pickle.dump(factor_value_adj2, f, -1)
f.close()
f = open(data_pat + '/fac_expand/other_facadj_3_20170101-20210228.pkl', 'wb')
pickle.dump(factor_value_adj3, f, -1)
f.close()
f = open(data_pat + '/fac_expand/other_facadj_4_20170101-20210228.pkl', 'wb')
pickle.dump(factor_value_adj4, f, -1)
f.close()
f = open(data_pat + '/fac_expand/other_facadj_5_20170101-20210228.pkl', 'wb')
pickle.dump(factor_value_adj5, f, -1)
f.close()
f = open(data_pat + '/fac_expand/other_facadj_6_20170101-20210228.pkl', 'wb')
pickle.dump(factor_value_adj6, f, -1)
f.close()

"""
# 对other_fac_all中的每个因子，将其划入到与自己的某类因子相关性最高的那类中
mine_data = pd.read_pickle(data_pat + '/fac_last.pkl')
all_data = dict(mine_data, **factor_value)
all_data = {k: v[(v.index >= '2017-01-01') & (v.index <= '2020-08-31')] for k, v in all_data.items()}
# 基础函数，计算因子之间的相关系数
def cal_factor_corr(fac_dict, pat_str):
    if not os.path.exists(pat_str):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(pat_str)
    total_data = pd.concat(fac_dict.values(), keys=fac_dict.keys())
    total_data = total_data.reset_index().set_index('level_1')
    corank_total = total_data.groupby(total_data.index).apply(lambda g: g.set_index('level_0').T.corr('spearman'))
    co_rank = corank_total.groupby(corank_total.index.get_level_values(1)).mean()
    co_rank = co_rank.reindex(co_rank.columns)  # 调整顺序，化为对称阵
    co_rank.to_csv(pat_str + "/rank_corr.csv", index=True, encoding='utf_8_sig')
    return co_rank
cal_factor_corr(all_data, data_pat + '/fac_expand')
"""
