import pandas as pd
import os
import pickle
from ft_platform.utils import utils_calculation as uc

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/fundamental'
fac_meaning = pd.read_excel(data_pat + '/all_fundamental.xlsx', sheet_name='基本面因子', index_col=0)
fac_growth = pd.read_pickle(data_pat + '/growth/factor_growth.pkl')
fac_earning = pd.read_pickle(data_pat + '/earning/factor_earning.pkl')
fac_valuation = pd.read_pickle(data_pat + '/valuation/fac_valuation.pkl')
# 只挑出夏普比率最高的前50%
fac_meaning = fac_meaning[fac_meaning['sharp_ratio'].rank(pct=True) >= 0.5]

"""
fac_meaning = fac_meaning[fac_meaning['tag1'] == 'growth']
fac_comb = {}
temp = {}
for tag in fac_meaning.index:
    temp[tag[:-3]] = uc.cs_rank(fac_growth[tag[:-3]])
comb = pd.concat(temp.values())
fac_comb['50%_eq_fundamental_growth'] = comb.groupby(comb.index).mean()
fac_comb['50%_eq_fundamental_growth'].index = pd.to_datetime(fac_comb['50%_eq_fundamental_growth'].index)
f = open(data_pat + '/50%_eq/fac_growth.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
"""
fac_meaning = fac_meaning[fac_meaning['tag1'] == 'earning']
fac_comb = {}
temp = {}
for tag in fac_meaning.index:
    temp[tag[:-3]] = uc.cs_rank(fac_earning[tag[:-3]])
comb = pd.concat(temp.values())
fac_comb['50%_eq_fundamental_earning'] = comb.groupby(comb.index).mean()
fac_comb['50%_eq_fundamental_earning'].index = pd.to_datetime(fac_comb['50%_eq_fundamental_earning'].index)
f = open(data_pat + '/50%_eq/fac_earning.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
"""
# """
fac_meaning = fac_meaning[fac_meaning['tag1'] == 'valuation']
fac_comb = {}
temp = {}
for tag in fac_meaning.index:
    temp[tag[:-3]] = uc.cs_rank(fac_valuation[tag[:-3]])
comb = pd.concat(temp.values())
fac_comb['50%_eq_fundamental_valuation'] = comb.groupby(comb.index).mean()
fac_comb['50%_eq_fundamental_valuation'].index = pd.to_datetime(fac_comb['50%_eq_fundamental_valuation'].index)
f = open(data_pat + '/50%_eq/fac_valuation.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
# """
