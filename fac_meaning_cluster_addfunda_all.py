import pandas as pd
import os
import h5py
import json
from ft_platform.utils import utils_calculation as uc
import itertools
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt
import pickle
from ft_platform.factor_process import fetch
from utils_func import query_data
from itertools import combinations

data_pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/all_cluster'
fac_meaning = pd.read_excel(data_pat + '/fac_addfunda/all_addfunda.xlsx', sheet_name='各类聚合因子的表现', index_col=0)

fac_old = pd.read_pickle(data_pat + '/fac_last.pkl')
fac_fundamental = {}
pat = 'E:/FT_Users/LihaiYang/Files/factor_comb_data/fac_meaning/fundamental/'
for way in ['all_eq', '50%_eq', 'sharpe_weight']:
    for j in os.listdir(pat + way):
        if j[-3:] == 'pkl':
            temp = pd.read_pickle(pat + way + '/' + j)
            fac_fundamental = dict(fac_fundamental, **temp)
fac_fundamental = {k: v for k, v in fac_fundamental.items() if k in ['50%_eq_fundamental_growth',
                                                                     '50%_eq_fundamental_earning',
                                                                     'sharpe_weight_fundamental_valuation']}
