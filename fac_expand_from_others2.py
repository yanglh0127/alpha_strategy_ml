import pandas as pd
import os
import json
from utils_func import query_data
from ft_platform.factor_process import fetch
import time
from ft_platform.utils import utils_calculation as uc
import pickle

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'

mine_data = pd.read_pickle('E:/FT_Users/LihaiYang/Files/factor_comb_data/all_fac_20170101-20210228/all_fac_20170101-20210228.pkl')
other_data1 = pd.read_pickle(data_pat + '/fac_expand/other_facadj_1_20170101-20210228.pkl')
other_data2 = pd.read_pickle(data_pat + '/fac_expand/other_facadj_2_20170101-20210228.pkl')
other_data3 = pd.read_pickle(data_pat + '/fac_expand/other_facadj_3_20170101-20210228.pkl')
other_data4 = pd.read_pickle(data_pat + '/fac_expand/other_facadj_4_20170101-20210228.pkl')
other_data5 = pd.read_pickle(data_pat + '/fac_expand/other_facadj_5_20170101-20210228.pkl')
other_data6 = pd.read_pickle(data_pat + '/fac_expand/other_facadj_6_20170101-20210228.pkl')
