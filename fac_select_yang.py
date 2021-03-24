import pandas as pd
from utils_func import query_data
from ft_platform.utils import utils_calculation as uc
import time

sort_var = 'sharp_ratio'  # 记得修改
select_num = 100  # 记得修改

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_20170101-20210228'  # 这边路径记得改
fac_data = pd.read_pickle(data_pat + '/all_fac_20170101-20210228.pkl')  # 记得修改
perf = pd.read_excel(data_pat + '/perf_summary_eq_tvwap.xlsx', index_col=0)  # 记得修改

