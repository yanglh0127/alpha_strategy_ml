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

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/5group/linear_model'  # 记得修改

begin = '2015-01-01'  # 记得修改
end = '2021-03-31'
trade_days = query_data.get_trade_days('d', from_trade_day=begin, to_trade_day=end)

# 读取最后选取的因子文件和对应的权重
with open(data_pat + "/fac_chosen.json",'r') as f:
    fac_choose = json.load(f)

# 因子分组情况和所有的因子list
factor_group = {}
factor_list = []
for k, v in fac_choose.items():
    temp = []
    for m, n in v.items():
        temp.extend(list(n.keys()))
    factor_group[k] = list(set(temp))
    factor_list.extend(list(set(temp)))

"""
# 读取因子数据  #183,183,168, price_reversal: 86; volume_std: 55; vp_corr: 9; financial report: 12; financial forword: 22
print('fetch')
fac_data = fetch.fetch_factor(begin, end, fields=list(set(factor_list)), standard='clean1_alla', codes=None, df=False)
f = open(data_pat + '/fac.pkl', 'wb')
pickle.dump(fac_data, f, -1)
f.close()
"""

fac_data = pd.read_pickle(data_pat + '/fac.pkl')
fac_data = {k: v for k, v in fac_data.items() if len(v) == len(trade_days)}  # 去掉数据缺失的因子

# top2000股票池
cap_data = fetch_data.fetch(begin, end, ['stock_tcap'])
cap_rank = cap_data['stock_tcap'].rank(axis=1, ascending=False)
# 每日的top2000股票标记为1，否则为nan
top2000 = (cap_rank <= 2000).where((cap_rank <= 2000) == 1)  # 2015年8月6日只有1999只?

# 根据top2000股票池把因子值在非2000的置为空值
fac_data = {k: (v * top2000) for k, v in fac_data.items()}

# 计算未来1、3、5、10、20日收益率，以开盘1小时tvwap为标准
end1 = '2021-04-30'
data = fetch_data.fetch(begin, end1, ['stock_adjtwap_0930_1030'])
index_data = fetch_data.fetch(begin, end1, ['index_close'], '000905')
stock_re = {}
stock_re['1_d'] = uc.ts_delay(data['stock_adjtwap_0930_1030'], -2) / uc.ts_delay(data['stock_adjtwap_0930_1030'], -1) - 1
stock_re['3_d'] = uc.ts_delay(data['stock_adjtwap_0930_1030'], -4) / uc.ts_delay(data['stock_adjtwap_0930_1030'], -1) - 1
stock_re['5_d'] = uc.ts_delay(data['stock_adjtwap_0930_1030'], -6) / uc.ts_delay(data['stock_adjtwap_0930_1030'], -1) - 1
stock_re['10_d'] = uc.ts_delay(data['stock_adjtwap_0930_1030'], -11) / uc.ts_delay(data['stock_adjtwap_0930_1030'], -1) - 1
stock_re['20_d'] = uc.ts_delay(data['stock_adjtwap_0930_1030'], -21) / uc.ts_delay(data['stock_adjtwap_0930_1030'], -1) - 1
stock_re = {k: v.loc[trade_days] for k, v in stock_re.items()}
index_re = {}
index_re['1_d'] = uc.ts_delay(index_data['index_close'], -1) / index_data['index_close'] - 1
index_re['3_d'] = uc.ts_delay(index_data['index_close'], -3) / index_data['index_close'] - 1
index_re['5_d'] = uc.ts_delay(index_data['index_close'], -5) / index_data['index_close'] - 1
index_re['10_d'] = uc.ts_delay(index_data['index_close'], -10) / index_data['index_close'] - 1
index_re['20_d'] = uc.ts_delay(index_data['index_close'], -20) / index_data['index_close'] - 1
index_re = {k: v.loc[trade_days] for k, v in index_re.items()}

index_re_n = {k: pd.DataFrame(np.tile(v.values, (1, len(list(stock_re[k])))),
                              index=stock_re[k].index, columns=list(stock_re[k])) for k, v in index_re.items()}

fac_data['stock_rela'] = stock_re['10_d'] - index_re_n['10_d']  # 记得修改

# 将因子数据和收益率数据合并
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
new_f = {}
for k, v in fac_data.items():
    new_v = pd.DataFrame(v.stack())
    new_v.columns = [k]
    new_f[k] = new_v
new_f = pd.concat(new_f.values(), axis=1)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
f = open(data_pat + '/new_f.pkl', 'wb')
pickle.dump(new_f, f, -1)
f.close()
