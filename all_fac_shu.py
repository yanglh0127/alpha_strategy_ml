import pandas as pd
from ft_platform.factor_process import fetch
import os
import h5py
from ft_platform.utils import utils_calculation as uc
from utils_func import query_data
import pickle

begin = '2017-01-01'
end = '2021-02-28'

# 读取舒昶因子表现汇总 # user = 'wangfeng:847','yyzhao:304','JiahuLi:191','Alex:195','cshu:97'
mine_summary = query_data.get_alphafactors_info(user='cshu')
print(len(mine_summary))
# 提取因子名
fac_name = [i['factor_name'] for i in mine_summary]
print(len(fac_name))
