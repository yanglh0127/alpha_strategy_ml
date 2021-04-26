import pandas as pd
import os
import pickle
from ft_platform.utils import utils_calculation as uc

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/fundamental'
fac_meaning = pd.read_excel(data_pat + '/all_fundamental.xlsx', sheet_name='基本面因子', index_col=0)
fac_growth = pd.read_pickle(data_pat + '/growth/factor_growth.pkl')
fac_earning = pd.read_pickle(data_pat + '/earning/factor_earning.pkl')
fac_valuation = pd.read_pickle(data_pat + '/valuation/fac_valuation.pkl')

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


"""
cal_factor_corr(fac_growth, data_pat + '/growth')
fac_comb = {}
temp = {}
for tag in fac_growth.keys():
    temp[tag] = uc.cs_rank(fac_growth[tag])
comb = pd.concat(temp.values())
fac_comb['all_eq_fundamental_growth'] = comb.groupby(comb.index).mean()
fac_comb['all_eq_fundamental_growth'].index = pd.to_datetime(fac_comb['all_eq_fundamental_growth'].index)
f = open(data_pat + '/all_eq/fac_growth.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
"""
cal_factor_corr(fac_earning, data_pat + '/earning')
fac_comb = {}
temp = {}
for tag in fac_earning.keys():
    temp[tag] = uc.cs_rank(fac_earning[tag])
comb = pd.concat(temp.values())
fac_comb['all_eq_fundamental_earning'] = comb.groupby(comb.index).mean()
fac_comb['all_eq_fundamental_earning'].index = pd.to_datetime(fac_comb['all_eq_fundamental_earning'].index)
f = open(data_pat + '/all_eq/fac_earning.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""

cal_factor_corr(fac_valuation, data_pat + '/valuation')
fac_comb = {}
temp = {}
for tag in fac_valuation.keys():
    temp[tag] = uc.cs_rank(fac_valuation[tag])
comb = pd.concat(temp.values())
fac_comb['all_eq_fundamental_valuation'] = comb.groupby(comb.index).mean()
fac_comb['all_eq_fundamental_valuation'].index = pd.to_datetime(fac_comb['all_eq_fundamental_valuation'].index)
f = open(data_pat + '/all_eq/fac_valuation.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
