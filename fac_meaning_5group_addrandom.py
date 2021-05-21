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

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/5group'  # 记得修改

begin = '2015-01-01'  # 记得修改
end = '2021-03-31'

fac_data = pd.read_pickle(data_pat + '/num_restrict/fac_comb.pkl')
noise = pd.DataFrame(np.random.standard_normal(fac_data['fac_choose_comb'].shape),
                     index=fac_data['fac_choose_comb'].index, columns=list(fac_data['fac_choose_comb']))

# top2000股票池
cap_data = fetch_data.fetch(begin, end, ['stock_tcap'])
cap_rank = cap_data['stock_tcap'].rank(axis=1, ascending=False)
# 每日的top2000股票标记为1，否则为nan
top2000 = (cap_rank <= 2000).where((cap_rank <= 2000) == 1)  # 2015年8月6日只有1999只?

# 根据top2000股票池把因子值在非2000的置为空值
noise = noise * top2000

fac_rand = {}
fac_rand['rand'] = noise.rank(axis=1)
fac_rand['info+rand'] = fac_data['fac_choose_comb'] + noise.rank(axis=1)
fac_rand['info+rand_eq'] = fac_data['fac_choose_comb'].rank(axis=1) + noise.rank(axis=1)
for k, v in fac_rand.items():
    a = v.notna().sum(axis=1)
    print(k, a.min(), a.max())

f = open(data_pat + '/add_random/fac_comb.pkl', 'wb')
pickle.dump(fac_rand, f, -1)
f.close()
