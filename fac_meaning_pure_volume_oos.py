from ft_platform import fetch_data
from ft_platform.utils import utils_calculation as uc
import pandas as pd
import pickle
from utils_func import query_data
from ft_platform.factor_process import fetch
import json
from copy import deepcopy
import numpy as np
import time
import json

begin = '2015-01-01'
end = '2020-12-31'

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/pure_volume'  # 记得修改

# 读取最后选取的因子文件和对应的权重
with open(data_pat + "/temp/fac_chosen.json",'r') as f:  # 记得修改
    fac_choose = json.load(f)
fac_choose = [(k, v) for k, v in fac_choose.items() if v != 0]
print(fac_choose)
factor_list = {fa[0]: fa[1] for fa in fac_choose}

# 提取因子数据
print('fetch')
fac_data = fetch.fetch_factor(begin, end, fields=list(factor_list.keys()), standard='clean1_alla', codes=None, df=False)

# top2000股票池
cap_data = fetch_data.fetch(begin, end, ['stock_tcap'])
cap_rank = cap_data['stock_tcap'].rank(axis=1, ascending=False)
# 每日的top2000股票标记为1，否则为nan
top2000 = (cap_rank <= 2000).where((cap_rank <= 2000) == 1)  # 2015年8月6日只有1999只?

# 根据top2000股票池把因子值在非2000的置为空值
fac_data = {k: (v * top2000) for k, v in fac_data.items()}
fac_data = {(k, v): (v * fac_data[k]) for k, v in factor_list.items()}

# 生成最终因子
fac_comb = fac_data[fac_choose[0]].rank(axis=1)
for k in fac_choose[1:]:
    fac_comb = fac_comb + fac_data[k].rank(axis=1)
a = fac_comb.notna().sum(axis=1)
print(a.min())
print(a.max())
fac_comb = {'fac_choose_comb': fac_comb}
f = open(data_pat + '/temp/fac_comb.pkl', 'wb')  # 记得修改
pickle.dump(fac_comb, f, -1)
f.close()
